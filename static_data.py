BRAWLERS = ['SHELLY', 'COLT', 'BULL', 'BROCK', 'RICO', 'SPIKE', 'BARLEY',
            'JESSIE', 'NITA', 'DYNAMIKE', 'EL PRIMO', 'MORTIS', 'CROW',
            'POCO', 'BO', 'PIPER', 'PAM', 'TARA', 'DARRYL', 'PENNY',
            'FRANK', 'GENE', 'TICK', 'LEON', 'ROSA', 'CARL', 'BIBI',
            '8-BIT', 'SANDY', 'BEA', 'EMZ', 'MR. P', 'MAX', 'JACKY',
            'GALE', 'NANI', 'SPROUT', 'SURGE', 'COLETTE', 'AMBER',
            'LOU', 'BYRON', 'EDGAR', 'RUFFS', 'STU', 'BELLE', 'SQUEAK',
            'GROM', 'BUZZ', 'GRIFF', 'ASH', 'MEG', 'LOLA', 'FANG',
            'EVE', 'JANET', 'BONNIE', 'OTIS', 'SAM', 'GUS', 'BUSTER',
            'CHESTER', 'GRAY', 'MANDY', 'R-T', 'WILLOW', 'MAISIE',
            'HANK', 'CORDELIUS', 'DOUG', 'PEARL','CHUCK', 'CHARLIE',
            'MICO', 'KIT', 'LARRY & LAWRIE', 'MELODIE', 'ANGELO',
            'DRACO', 'LILY', 'BERRY', 'CLANCY', 'MOE', 'KENJI',
            'SHADE', 'JUJU', 'MEEPLE', 'OLLIE', 'LUMI', 'FINX',
            'JAE-YONG', 'KAZE', 'ALLI', 'TRUNK', 'MINA']

MODES = ['bounty', 'hotZone', 'knockout', 'brawlBall', 'heist', 'gemGrab']

MAPS = ["Belle's Rock", 'Bridge Too Far', 'Center Stage',
        'Double Swoosh', 'Dry Season', 'Dueling Beetles', 'Excel', 'Flaring Phoenix',
        'Flowing Springs', 'Gem Fort', 'Goldarm Gulch', 'Hard Rock Mine',
        'Hideout', 'Hot Potato', 'Infinite Doom', 'Kaboom Canyon', 'Layer Cake', 'New Horizons',
        'Open Business', 'Out in the Open', 'Parallel Plays', 'Pinball Dreams',
        'Ring of Fire', 'Safe Zone', 'Shooting Star', 'Sneaky Fields', 'Triple Dribble',
        'Undermine']

MODE_LEN = 10
MAP_LEN = 50
BRAWLER_LEN = 110

MAPS_FOR_MODES = {
    'bounty': [
        "Hideout", "Shooting Star", "Dry Season", "Layer Cake", 'Excel', 'Infinite Doom'
    ],
    'hotZone': [
        "Open Business", "Parallel Plays", "Ring of Fire", "Dueling Beetles"
    ],
    'knockout': [
        "Belles Rock", "Flaring Phoenix", "Goldarm Gulch", "Out in the Open", "Flowing Springs", "New Horizons"
    ],
    'brawlBall': [
        "Center Stage", "Triple Dribble", "Pinball Dreams", "Sneaky Fields"
    ],
    'heist': [
        "Kaboom Canyon", "Safe Zone", "Bridge Too Far", "Hot Potato",
    ],
    'gemGrab': [
        "Hard Rock Mine", "Gem Fort", "Double Swoosh", "Undermine"
    ]
}

MODES_FOR_MAPS = {map_name: mode for mode, maps in MAPS_FOR_MODES.items() for map_name in maps}

RESULTS = ['victory', 'defeat', 'draw']

MAPS_RU = {
    "Belle's Rock": "Живописный утес",
    "Bridge Too Far": "Взятие моста",
    "Center Stage": "В центре внимания",
    "Double Swoosh": "Вжух-вжух",
    "Dry Season": "Засуха",
    "Dueling Beetles": "Муравьиные бега",
    "Flaring Phoenix": "Пылающий Феникс",
    "Flowing Springs": "Бурные ключи",
    "Gem Fort": "Кристальный форт",
    "Goldarm Gulch": "Ущелье золотой руки",
    "Hard Rock Mine": "Роковая шахта",
    "Hideout": "Укрытие",
    "Hot Potato": "Горячая кукуруза",
    "Kaboom Canyon": "Пороховой каньон",
    "Layer Cake": "Слоёный торт",
    "New Horizons": "Новые горизонты",
    "Open Business": "Открыто",
    "Out in the Open": "В чистом поле",
    "Parallel Plays": "Параллельная игра",
    "Pinball Dreams": "Пинбол",
    "Ring of Fire": "Огненное кольцо",
    "Safe Zone": "Надежное укрытие",
    "Shooting Star": "Падающая звезда",
    "Sneaky Fields": "Зловредные поля",
    "Triple Dribble": "Трипл-дриблинг",
    "Undermine": "Подрыв"
}


VARIANTS = {
    "8-BIT": ["8-бит", "8-бит", "8бит", "эйтбит", "эйт бит"],
    "ALLI": ["алли"],
    "AMBER": ["амбер", "эмбер"],
    "ANGELO": ["анджело", "анжело"],
    "ASH": ["эш", "аш"],
    "BARLEY": ["барли", "барлей"],
    "BEA": ["беа", "бэа", "би", "пчела"],
    "BELLE": ["белль"],
    "BERRY": ["берри"],
    "BIBI": ["биби"],
    "BO": ["бо"],
    "BONNIE": ["бонни"],
    "BROCK": ["брок"],
    "BULL": ["булл", "бык"],
    "BUSTER": ["бастер"],
    "BUZZ": ["базз"],
    "BYRON": ["байрон"],
    "CARL": ["карл"],
    "CHARLIE": ["чарли"],
    "CHESTER": ["честер"],
    "CHUCK": ["чак"],
    "CLANCY": ["кленси", "клэнси"],
    "COLETTE": ["колетт", "коллет"],
    "COLT": ["кольт"],
    "CORDELIUS": ["корнелиус", "корделиус", "корд"],
    "CROW": ["ворон", "ворона", "кроу"],
    "DARRYL": ["дэррил"],
    "DOUG": ["даг", "доуг", "сосиска"],
    "DRACO": ["драко"],
    "DYNAMIKE": ["дина", "динамайк", "дед"],
    "EDGAR": ["эдгар", "эдгор", "шарфик", "шарф"],
    "EL PRIMO": ["эльпримо", "эль примо", "примо", "примат"],
    "EMZ": ["эмз", "емз"],
    "EVE": ["ева", "блоха"],
    "FANG": ["фэнг", "фенг"],
    "FINX": ["финкс"],
    "FRANK": ["фрэнк"],
    "GALE": ["гэйл", "гейл"],
    "GENE": ["джин"],
    "GRAY": ["грей", "грэй"],
    "GRIFF": ["грифф"],
    "GROM": ["гром"],
    "GUS": ["гас", "гусь"],
    "HANK": ["хэнк", "хенк", "рыба"],
    "JACKY": ["джеки", "джекки"],
    "JAE-YONG": ["чжэ-ён", "чжэ-ен", "джэ ён", "джэ-ён", "дже-енг", "джейонг"],
    "JANET": ["джанет"],
    "JESSIE": ["джесси"],
    "JUJU": ["джуджу", "джу джу", "джу-джу"],
    "KAZE": ["кадзэ", "казе", "кадзе"],
    "KENJI": ["кэндзи", "кенджи", "кенши"],
    "KIT": ["кит", "кот"],
    "LARRY & LAWRIE": ["ларри и лорри", "лил", "ларри"],
    "LEON": ["леон", "лэон"],
    "LILY": ["лили"],
    "LOLA": ["лола"],
    "LOU": ["лу"],
    "LUMI": ["луми", "люми"],
    "MAISIE": ["мейси", "мэйси"],
    "MANDY": ["мэнди", "менди"],
    "MAX": ["макс"],
    "MEEPLE": ["мипл"],
    "MEG": ["мэг", "мег"],
    "MELODIE": ["мелоди", "мэлоди"],
    "MICO": ["мико", "макака"],
    "MOE": ["мо"],
    "MORTIS": ["мортис"],
    "MR. P": ["мистер пи", "мистер п", "м пи"],
    "NANI": ["нани"],
    "NITA": ["нита"],
    "OLLIE": ["олли"],
    "OTIS": ["отис"],
    "PAM": ["пэм", "пем"],
    "PEARL": ["перл", "пёрл"],
    "PENNY": ["пенни", "пэнни"],
    "PIPER": ["пайпер"],
    "POCO": ["поко"],
    "R-T": ["рт", "ар-ти"],
    "RICO": ["рико", "рикошет"],
    "ROSA": ["роза"],
    "RUFFS": ["рафс", "гавс", "собака", "генерал гавс"],
    "SAM": ["сэм", "сем"],
    "SANDY": ["сэнди", "сенди"],
    "SHADE": ["шейд", "шэйд"],
    "SHELLY": ["шелли", "шэлли"],
    "SPIKE": ["спайк", "кактус"],
    "SPROUT": ["спраут"],
    "SQUEAK": ["сквик", "скуик"],
    "STU": ["стю", "сту"],
    "SURGE": ["вольт"],
    "TARA": ["тара"],
    "TICK": ["тик"],
    "TRUNK": ["транк"],
    "WILLOW": ["уиллоу", "виллоу"],
}
