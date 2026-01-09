player_prompts = {
    "en": """You are Player {player_id}. You are playing the 15-Puzzle game.\n\
            The objective of the game is to arrange the numbered tiles in ascending order from 1 to 15, with the empty space located in the bottom-right corner.\n\
            To make a move, you can slide a tile into the empty space (represented by a double underscore, e.g. __) by using one of the following commands:\n\
            - 'up': Move the tile below the empty space up.\n\
            - 'down': Move the tile above the empty space down.\n\
            - 'left': Move the tile to the right of the empty space left.\n\
            - 'right': Move the tile to the left of the empty space right.\n\
            To submit your move, type the direction (e.g., 'up', 'down', 'left', or 'right') in square brackets, e.g. [up].\n\
            The current board layout is shown below. Use the information to solve the puzzle.\n""",
    
    "hi": """आप खिलाड़ी {player_id} हैं। आप 15-पज़ल खेल खेल रहे हैं।\n\
            खेल का उद्देश्य संख्यांकित टाइलों को 1 से 15 तक आरोही क्रम में व्यवस्थित करना है, जिसमें खाली स्थान निचले-दाएँ कोने में हो।\n\
            चाल चलने के लिए, आप निम्नलिखित आदेशों में से किसी एक का उपयोग करके किसी टाइल को खाली स्थान (जो डबल अंडरस्कोर द्वारा दर्शाया गया है, जैसे __) में खिसका सकते हैं:\n\
            - 'up': खाली स्थान के नीचे की टाइल को ऊपर ले जाएँ।\n\
            - 'down': खाली स्थान के ऊपर की टाइल को नीचे ले जाएँ।\n\
            - 'left': खाली स्थान के दाएँ की टाइल को बाएँ ले जाएँ।\n\
            - 'right': खाली स्थान के बाएँ की टाइल को दाएँ ले जाएँ।\n\
            अपनी चाल जमा करने के लिए, दिशा (जैसे, 'up', 'down', 'left', या 'right') को वर्ग कोष्ठकों में टाइप करें, जैसे [up]।\n\
            वर्तमान बोर्ड लेआउट नीचे दिखाया गया है। पहेली को हल करने के लिए जानकारी का उपयोग करें।\n""",
    
    "zh": """你是玩家 {player_id}。你正在玩15拼图游戏。\n\
            游戏的目标是将编号的方块按从1到15的升序排列，空格位于右下角。\n\
            要进行移动，你可以使用以下命令之一，将一个方块滑入空格（由双下划线表示，例如 __）：\n\
            - 'up': 将空格下方的方块上移。\n\
            - 'down': 将空格上方的方块下移。\n\
            - 'left': 将空格右侧的方块左移。\n\
            - 'right': 将空格左侧的方块右移。\n\
            要提交你的移动，请在方括号中输入方向（例如 'up'、'down'、'left' 或 'right'），如 [up]。\n\
            当前棋盘布局如下。使用这些信息来解开拼图。\n""",
    
    "es": """Eres el jugador {player_id}. Estás jugando al juego del Rompecabezas 15.\n\
            El objetivo del juego es ordenar las fichas numeradas en orden ascendente del 1 al 15, con el espacio vacío ubicado en la esquina inferior derecha.\n\
            Para hacer un movimiento, puedes deslizar una ficha al espacio vacío (representado por un doble guion bajo, por ejemplo __) usando uno de los siguientes comandos:\n\
            - 'up': Mueve hacia arriba la ficha debajo del espacio vacío.\n\
            - 'down': Mueve hacia abajo la ficha encima del espacio vacío.\n\
            - 'left': Mueve hacia la izquierda la ficha a la derecha del espacio vacío.\n\
            - 'right': Mueve hacia la derecha la ficha a la izquierda del espacio vacío.\n\
            Para enviar tu movimiento, escribe la dirección (por ejemplo, 'up', 'down', 'left' o 'right') entre corchetes, por ejemplo [up].\n\
            El diseño actual del tablero se muestra a continuación. Usa la información para resolver el rompecabezas.\n""",
    
    "de": """Du bist Spieler {player_id}. Du spielst das 15-Puzzle-Spiel.\n\
            Das Ziel des Spiels ist es, die nummerierten Kacheln in aufsteigender Reihenfolge von 1 bis 15 anzuordnen, wobei das leere Feld in der unteren rechten Ecke liegt.\n\
            Um einen Zug zu machen, kannst du eine Kachel in das leere Feld (dargestellt durch zwei Unterstriche, z. B. __) schieben, indem du einen der folgenden Befehle verwendest:\n\
            - 'up': Bewege die Kachel unter dem leeren Feld nach oben.\n\
            - 'down': Bewege die Kachel über dem leeren Feld nach unten.\n\
            - 'left': Bewege die Kachel rechts vom leeren Feld nach links.\n\
            - 'right': Bewege die Kachel links vom leeren Feld nach rechts.\n\
            Um deinen Zug einzureichen, tippe die Richtung (z. B. 'up', 'down', 'left' oder 'right') in eckige Klammern, z. B. [up].\n\
            Das aktuelle Spielfeldlayout wird unten angezeigt. Verwende die Informationen, um das Puzzle zu lösen.\n""",
    
    "sw": """Wewe ni mchezaji {player_id}. Unacheza mchezo wa Fumbo la 15.\n\
            Lengo la mchezo ni kupanga vigae vyenye namba kwa mpangilio wa kupanda kutoka 1 hadi 15, huku nafasi tupu ikiwa kona ya chini-kulia.\n\
            Ili kufanya hatua, unaweza kusogeza jiwe moja kwenye nafasi tupu (inayowakilishwa na mstari miwili ya chini, mfano __) kwa kutumia mojawapo ya amri zifuatazo:\n\
            - 'up': Sogeza jiwe lililo chini ya nafasi tupu juu.\n\
            - 'down': Sogeza jiwe lililo juu ya nafasi tupu chini.\n\
            - 'left': Sogeza jiwe lililo kulia mwa nafasi tupu kushoto.\n\
            - 'right': Sogeza jiwe lililo kushoto mwa nafasi tupu kulia.\n\
            Ili kuwasilisha hatua yako, andika mwelekeo (mfano, 'up', 'down', 'left', au 'right') ndani ya mabano ya mraba, mfano [up].\n\
            Mpangilio wa sasa wa ubao umeonyeshwa hapa chini. Tumia habari hiyo kutatua fumbo.\n""",
    
    "ru": """Вы игрок {player_id}. Вы играете в игру «Пятнашки».\n\
            Цель игры — расположить пронумерованные плитки в порядке возрастания от 1 до 15, оставив пустую ячейку в правом нижнем углу.\n\
            Чтобы сделать ход, вы можете сдвинуть плитку в пустое место (обозначенное двойным подчеркиванием, например __) с помощью одной из следующих команд:\n\
            - 'up': Передвиньте плитку под пустой ячейкой вверх.\n\
            - 'down': Передвиньте плитку над пустой ячейкой вниз.\n\
            - 'left': Передвиньте плитку справа от пустой ячейки влево.\n\
            - 'right': Передвиньте плитку слева от пустой ячейки вправо.\n\
            Чтобы отправить ход, введите направление (например, 'up', 'down', 'left' или 'right') в квадратных скобках, например [up].\n\
            Текущая раскладка поля показана ниже. Используйте эту информацию, чтобы решить головоломку.\n""",
    
    "te": """మీరు ఆటగాడు {player_id}. మీరు 15-పజిల్ ఆట ఆడుతున్నారు.\n\
            ఈ ఆటలో లక్ష్యం సంఖ్యల టైల్స్‌ను 1 నుండి 15 వరకూ పెరుగుతున్న క్రమంలో అమర్చడం, ఖాళీ స్థలం కుడి-క్రింది మూలలో ఉండాలి.\n\
            ఒక కదలిక చేయడానికి, మీరు టైల్ను ఖాళీ స్థలంలోకి జరపవచ్చు (దీనిని డబుల్ అండర్‌స్కోర్‌తో సూచిస్తారు, ఉదా. __) క్రింది ఆదేశాలలో ఏదో ఒకదాన్ని ఉపయోగించి:\n\
            - 'up': ఖాళీ స్థలానికి కింద ఉన్న టైల్ను పైకి జరపండి.\n\
            - 'down': ఖాళీ స్థలానికి పైగా ఉన్న టైల్ను కిందికి జరపండి.\n\
            - 'left': ఖాళీ స్థలానికి కుడివైపు ఉన్న టైల్ను ఎడమకు జరపండి.\n\
            - 'right': ఖాళీ స్థలానికి ఎడమ వైపు ఉన్న టైల్ను కుడి వైపు జరపండి.\n\
            మీ కదలికను సమర్పించడానికి, దిశను (ఉదా. 'up', 'down', 'left', లేదా 'right') చతురస్ర కౌగిట్లలో టైప్ చేయండి, ఉదా. [up].\n\
            ప్రస్తుత బోర్డు అమరిక క్రింద చూపబడింది. పజిల్‌ని పరిష్కరించడానికి ఈ సమాచారాన్ని ఉపయోగించండి.\n""",
    
    "ja": """あなたはプレイヤー {player_id} です。あなたは15パズルゲームをプレイしています。\n\
            ゲームの目的は、番号の付いたタイルを1から15まで昇順に並べ、空白を右下の隅に配置することです。\n\
            動かすには、次のコマンドのいずれかを使用して、タイルを空白（ダブルアンダースコアで表される、例: __）にスライドさせます:\n\
            - 'up': 空白の下のタイルを上に動かす。\n\
            - 'down': 空白の上のタイルを下に動かす。\n\
            - 'left': 空白の右のタイルを左に動かす。\n\
            - 'right': 空白の左のタイルを右に動かす。\n\
            手を提出するには、方向（例: 'up'、'down'、'left'、'right'）を角括弧で入力してください。例: [up]。\n\
            現在のボードの配置は以下に示されています。情報を使ってパズルを解きましょう。\n"""
}
