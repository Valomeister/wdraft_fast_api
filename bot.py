import os
import cv2
import numpy as np
import telebot
import io

import MCTS
import static_data
from screenshot_decomposition import decompose_screenshot

BOT_TOKEN = "7528119037:AAFbGfckppcrKRX2FLsFPM1U87fI34pjkC4"
telebot.apihelper.ENABLE_MIDDLEWARE = True
bot = telebot.TeleBot(BOT_TOKEN)

# создаём папку, если её нет
os.makedirs("images", exist_ok=True)


def format_screenshot_info(screenshot_info):
    map = screenshot_info['map']['name']
    mode = static_data.MODES_FOR_MAPS[map]
    teamA = screenshot_info['picks']['team_blue']
    teamB = screenshot_info['picks']['team_red']

    match = {
        "mode": mode,
        "map": map,
        "teams": [
            teamA,
            teamB
        ]
    }

    bans_teamA = screenshot_info['bans']['team_blue']
    bans_teamB = screenshot_info['bans']['team_red']

    brawlers_set = set(bans_teamA + bans_teamB)
    bans_mask = np.zeros(96)
    for b in brawlers_set:
        if b in static_data.BRAWLERS:
            idx = static_data.BRAWLERS.index(b)
            bans_mask[idx] = 1

    return match, bans_mask


def get_readable_mcts_prediction(match, searches, bans_mask, n):
    probs, val = MCTS.get_mcts_results([match], searches, bans_mask)
    top_n = MCTS.top_n_brawlers(probs, n)
    readable = "Топ драфтов:\n" + "\n".join([f"{pos}. {static_data.BRAWLERS[idx]}" for pos, idx in top_n])
    return readable

user_states = {}

@bot.middleware_handler()
def notify_usage(bot_instance, update):
    if hasattr(update, "message") and update.message:
        message = update.message

        username = message.from_user.username
        if message.text:
            print(f"@{username} отправил в бота текст")
        elif message.photo:
            print(f"@{username} отправил в бота фото")
        else:
            print(f"@{username} отправил в бота нечто")


@bot.middleware_handler()
def add_user_if_needed(bot_instance, update):
    if hasattr(update, "message") and update.message:
        message = update.message

        user_id = message.from_user.id
        if not user_id in user_states:

            username = message.from_user.username
            user_states[user_id] = {
                "username": username,
                "awaiting_cur_turn_info": False,
                "match": None,
                "bans_mask": None
            }

            print("User created")



@bot.message_handler(content_types=['photo'])
def save_resized_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)

    # читаем изображение в OpenCV
    img_array = np.frombuffer(downloaded, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    h, w = img.shape[:2]
    target_width = 1280

    # ресайз по ширине с сохранением пропорций
    if w != target_width:
        ratio = target_width / w
        new_height = int(h * ratio)
        img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)

    filename = f"images/{file_id}.jpg"
    cv2.imwrite(filename, img)

    screenshot_info, debug_img = decompose_screenshot(filename)

    _, buffer = cv2.imencode(".jpg", debug_img)
    bytes_img = io.BytesIO(buffer)
    bot.send_photo(message.chat.id, bytes_img, caption="Отладка: декомпозиция скриншота")

    info_str = (
        "Информация со скриншота:\n"
        f"Карта: {screenshot_info['map']['name']} ({screenshot_info['map']['lang']})\n\n"
        "Пики:\n"
        f"{', '.join(screenshot_info['picks']['team_blue'])} против "
        f"{', '.join(screenshot_info['picks']['team_red'])}\n\n"
        "Баны:\n"
        f"{', '.join(screenshot_info['bans']['team_blue'])} против "
        f"{', '.join(screenshot_info['bans']['team_red'])}"
    )

    bot.send_message(message.chat.id, info_str)

    match, bans_mask = format_screenshot_info(screenshot_info)
    user_states[message.from_user.id]['match'] = match
    user_states[message.from_user.id]['bans_mask'] = bans_mask

    if len(match["teams"][0]) > len(match["teams"][1]):
        match["teams"][0], match["teams"][1] = match["teams"][1], match["teams"][0]

        preds = get_readable_mcts_prediction(user_states[message.from_user.id]['match'], 3000, bans_mask, 5)
        bot.send_message(message.chat.id, preds)

    elif len(match["teams"][0]) == len(match["teams"][1]):
        preds = get_readable_mcts_prediction(user_states[message.from_user.id]['match'], 3000, bans_mask, 5)
        bot.send_message(message.chat.id, preds)

        # # --- отправляем вопрос с кнопками ---
        # markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        # markup.add("Синих", "Красных")
        # question_msg = bot.send_message(message.chat.id,
        #                                 "Количество персонажей одинаково. Чей сейчас ход?",
        #                                 reply_markup=markup)
        #
        # # сохраняем состояние пользователя, чтобы знать, что ждем ответ
        # user_states[message.from_user.id]['awaiting_cur_turn_info'] = True
    else:
        preds = get_readable_mcts_prediction(user_states[message.from_user.id]['match'], 3000, bans_mask, 5)
        bot.send_message(message.chat.id, preds)

@bot.message_handler(func=lambda msg: True)
def handle_response(message):
    chat_id = message.chat.id
    text = message.text.strip().lower()

    if user_states[message.from_user.id]['awaiting_cur_turn_info']:
        match = user_states[message.from_user.id]['match']
        bans_mask = user_states[message.from_user.id]['bans_mask']

        if text == "красных":
            match['teams'][0], match['teams'][1] = match['teams'][1], match['teams'][0]

        preds = get_readable_mcts_prediction(match, 6000, bans_mask, 5)
        bot.send_message(message.chat.id, preds)

        user_states[message.from_user.id]['awaiting_cur_turn_info'] = False
    else:
        # обычный эхо или сообщение для всех остальных
        bot.send_message(chat_id, "Я не в ресурсе сори")




bot.infinity_polling()
