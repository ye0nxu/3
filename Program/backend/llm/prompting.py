from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


## =====================================
## 한국어 색상 → 영어 색상 매핑 테이블
## =====================================
_KOREAN_COLOR_MAP: dict[str, str] = {
    "흰": "white",
    "하얀": "white",
    "흰색": "white",
    "하얀색": "white",
    "화이트": "white",
    "파란": "blue",
    "파랑": "blue",
    "파란색": "blue",
    "푸른": "blue",
    "블루": "blue",
    "빨간": "red",
    "빨강": "red",
    "빨간색": "red",
    "빨강색": "red",
    "레드": "red",
    "검은": "black",
    "검정": "black",
    "검은색": "black",
    "검정색": "black",
    "까만": "black",
    "블랙": "black",
    "노란": "yellow",
    "노랑": "yellow",
    "노란색": "yellow",
    "노랑색": "yellow",
    "옐로": "yellow",
    "초록": "green",
    "초록색": "green",
    "녹색": "green",
    "그린": "green",
    "회색": "gray",
    "회색빛": "gray",
    "그레이": "gray",
    "은색": "silver",
    "실버": "silver",
    "주황": "orange",
    "주황색": "orange",
    "오렌지": "orange",
    "오렌지색": "orange",
    "보라": "purple",
    "보라색": "purple",
    "갈색": "brown",
    "브라운": "brown",
    "분홍": "pink",
    "분홍색": "pink",
    "핑크": "pink",
    "하늘색": "sky blue",
    "남색": "navy",
    "베이지": "beige",
    "아이보리": "ivory",
    "금색": "gold",
    "골드": "gold",
}

## =====================================
## 한국어 객체명 → 영어 프롬프트 매핑 테이블
## 우선순위: 더 긴(구체적인) 토큰이 먼저 위치하도록 정렬
## =====================================
_KOREAN_OBJECT_MAP: dict[str, str] = {
    # ── 자동차 조명 (가장 구체적 → 덜 구체적 순)
    "후미등": "tail light",
    "미등": "tail light",
    "테일램프": "tail light",
    "테일라이트": "tail light",
    "리어램프": "rear lamp",
    "리어라이트": "rear light",
    "전조등": "headlight",
    "헤드라이트": "headlight",
    "헤드램프": "headlight",
    "안개등": "fog light",
    "포그라이트": "fog light",
    "방향지시등": "turn signal",
    "깜빡이": "turn signal",
    "방향등": "turn signal",
    "브레이크등": "brake light",
    "제동등": "brake light",
    "후진등": "reverse light",
    "주간주행등": "daytime running light",
    "drl": "daytime running light",
    "실내등": "interior light",
    "번호판등": "license plate light",

    # ── 자동차 외장 부품
    "앞범퍼": "front bumper",
    "뒷범퍼": "rear bumper",
    "후방범퍼": "rear bumper",
    "범퍼": "bumper",
    "앞그릴": "front grille",
    "라디에이터그릴": "front grille",
    "그릴": "grille",
    "보닛": "hood",
    "후드": "hood",
    "트렁크": "trunk",
    "트렁크리드": "trunk lid",
    "루프": "roof",
    "차량루프": "car roof",
    "선루프": "sunroof",
    "윈도우": "window",
    "창문": "window",
    "앞유리": "windshield",
    "전면유리": "windshield",
    "뒷유리": "rear windshield",
    "측면유리": "side window",
    "사이드윈도우": "side window",
    "와이퍼": "wiper",
    "앞와이퍼": "front wiper",
    "뒤와이퍼": "rear wiper",
    "사이드미러": "side mirror",
    "백미러": "rearview mirror",
    "외부미러": "side mirror",
    "도어미러": "door mirror",
    "문": "door",
    "차문": "car door",
    "앞문": "front door",
    "뒷문": "rear door",
    "도어": "door",
    "도어핸들": "door handle",
    "손잡이": "handle",
    "휠": "wheel",
    "타이어": "tire",
    "휠아치": "wheel arch",
    "펜더": "fender",
    "앞펜더": "front fender",
    "쿼터패널": "quarter panel",
    "필러": "pillar",
    "a필러": "A-pillar",
    "b필러": "B-pillar",
    "c필러": "C-pillar",
    "사이드스텝": "side step",
    "스포일러": "spoiler",
    "리어스포일러": "rear spoiler",
    "에어댐": "air dam",
    "번호판": "license plate",
    "넘버플레이트": "license plate",
    "엠블럼": "emblem",
    "배지": "badge",
    "선바이저": "sun visor",
    "안테나": "antenna",

    # ── 자동차 내장
    "운전대": "steering wheel",
    "핸들": "steering wheel",
    "대시보드": "dashboard",
    "계기판": "instrument panel",
    "센터콘솔": "center console",
    "기어": "gear shift",
    "기어레버": "gear shift",
    "시트": "seat",
    "좌석": "seat",
    "운전석": "driver seat",
    "조수석": "passenger seat",
    "헤드레스트": "headrest",
    "안전벨트": "seatbelt",
    "에어백": "airbag",
    "페달": "pedal",

    # ── 차종
    "승용차": "car",
    "자동차": "car",
    "차량": "car",
    "차": "car",
    "트럭": "truck",
    "화물차": "truck",
    "버스": "bus",
    "밴": "van",
    "승합차": "van",
    "오토바이": "motorcycle",
    "모터사이클": "motorcycle",
    "이륜차": "motorcycle",
    "자전거": "bicycle",
    "킥보드": "scooter",
    "전동킥보드": "electric scooter",
    "suv": "SUV",
    "픽업트럭": "pickup truck",
    "덤프트럭": "dump truck",
    "콘크리트믹서": "cement mixer",
    "탱크로리": "tanker truck",
    "견인차": "tow truck",
    "굴삭기": "excavator",
    "포클레인": "excavator",
    "지게차": "forklift",
    "크레인": "crane",
    "불도저": "bulldozer",

    # ── 사람 / 보호장구
    "사람": "person",
    "남자": "man",
    "여자": "woman",
    "어린이": "child",
    "어른": "adult",
    "보행자": "pedestrian",
    "작업자": "worker",
    "안전모": "hard hat",
    "헬멧": "helmet",
    "안전조끼": "safety vest",
    "조끼": "vest",
    "안전복": "safety suit",
    "장갑": "glove",
    "안전화": "safety boot",
    "마스크": "mask",
    "고글": "goggles",
    "안전벨트": "harness",
    "보호대": "protective gear",

    # ── 도로 / 교통 시설
    "신호등": "traffic light",
    "교통신호등": "traffic light",
    "가로등": "street light",
    "표지판": "sign",
    "도로표지판": "road sign",
    "안내판": "sign board",
    "차선": "lane marking",
    "중앙선": "center line",
    "횡단보도": "crosswalk",
    "인도": "sidewalk",
    "가드레일": "guardrail",
    "방호울타리": "crash barrier",
    "콘": "traffic cone",
    "안전콘": "traffic cone",
    "라바콘": "traffic cone",
    "드럼": "drum barrel",
    "안전드럼": "safety drum",
    "펜스": "fence",
    "공사펜스": "construction fence",
    "볼라드": "bollard",
    "버스정류장": "bus stop",
    "주차구역": "parking space",
    "주차선": "parking line",

    # ── 동물
    "강아지": "dog",
    "개": "dog",
    "고양이": "cat",
    "새": "bird",
    "소": "cow",
    "말": "horse",
    "돼지": "pig",
    "양": "sheep",
    "닭": "chicken",
    "오리": "duck",
    "토끼": "rabbit",
    "곰": "bear",
    "호랑이": "tiger",
    "사자": "lion",
    "코끼리": "elephant",
    "기린": "giraffe",
    "원숭이": "monkey",
    "물고기": "fish",
    "상어": "shark",
    "고래": "whale",
    "독수리": "eagle",
    "앵무새": "parrot",
    "뱀": "snake",
    "도마뱀": "lizard",
    "개구리": "frog",
    "나비": "butterfly",
    "벌": "bee",
    "개미": "ant",
    "거미": "spider",
    "고양이 귀": "cat ear",
    "강아지 귀": "dog ear",
    "강아지 발": "dog paw",
    "고양이 발": "cat paw",
    "얼룩 고양이": "tabby cat",
    "줄무늬 고양이": "striped cat",
    "검은 고양이": "black cat",

    # ── 음식 / 식재료
    "라면": "ramen",
    "국수": "noodles",
    "밥": "rice",
    "볶음밥": "fried rice",
    "김밥": "kimbap",
    "초밥": "sushi",
    "빵": "bread",
    "케이크": "cake",
    "과자": "snack",
    "쿠키": "cookie",
    "피자": "pizza",
    "햄버거": "burger",
    "샌드위치": "sandwich",
    "샐러드": "salad",
    "스테이크": "steak",
    "고기": "meat",
    "생선": "fish",
    "달걀": "egg",
    "계란": "egg",
    "두부": "tofu",
    "치즈": "cheese",
    "버터": "butter",
    "사과": "apple",
    "빨간 사과": "red apple",
    "배": "pear",
    "포도": "grapes",
    "딸기": "strawberry",
    "바나나": "banana",
    "오렌지": "orange",
    "레몬": "lemon",
    "수박": "watermelon",
    "멜론": "melon",
    "복숭아": "peach",
    "체리": "cherry",
    "블루베리": "blueberry",
    "키위": "kiwi",
    "망고": "mango",
    "파인애플": "pineapple",
    "당근": "carrot",
    "양파": "onion",
    "감자": "potato",
    "고구마": "sweet potato",
    "브로콜리": "broccoli",
    "토마토": "tomato",
    "오이": "cucumber",
    "호박": "pumpkin",
    "가지": "eggplant",
    "버섯": "mushroom",
    "마늘": "garlic",
    "파": "green onion",
    "배추": "cabbage",
    "시금치": "spinach",
    "상추": "lettuce",
    "콩": "beans",
    "고추": "pepper",
    "파프리카": "bell pepper",
    "라면 그릇": "ramen bowl",
    "밥그릇": "rice bowl",
    "접시": "plate",
    "냄비": "pot",
    "프라이팬": "frying pan",

    # ── 음료
    "커피": "coffee",
    "커피잔": "coffee cup",
    "차": "tea",
    "녹차": "green tea",
    "주스": "juice",
    "물": "water",
    "물병": "water bottle",
    "음료수": "beverage",
    "맥주": "beer",
    "와인": "wine",
    "와인잔": "wine glass",
    "물컵": "glass",
    "컵": "cup",
    "머그컵": "mug",

    # ── 가구 / 실내
    "소파": "sofa",
    "침대": "bed",
    "책상": "desk",
    "의자": "chair",
    "식탁": "dining table",
    "테이블": "table",
    "탁자": "table",
    "책장": "bookshelf",
    "선반": "shelf",
    "장롱": "wardrobe",
    "옷장": "closet",
    "서랍": "drawer",
    "커튼": "curtain",
    "블라인드": "blinds",
    "카페트": "carpet",
    "러그": "rug",
    "액자": "picture frame",
    "거울": "mirror",
    "시계": "clock",
    "벽시계": "wall clock",
    "스탠드": "lamp",
    "조명": "light",
    "화분": "plant pot",
    "화병": "vase",
    "의자 다리": "chair leg",
    "나무 의자": "wooden chair",
    "소파 쿠션": "sofa cushion",

    # ── 건물 / 구조물
    "건물": "building",
    "창": "window",
    "출입문": "door",
    "현관": "entrance",
    "계단": "stairs",
    "엘리베이터": "elevator",
    "에스컬레이터": "escalator",
    "기둥": "pillar",
    "벽": "wall",
    "지붕": "roof",
    "굴뚝": "chimney",
    "교량": "bridge",
    "다리": "bridge",
    "터널": "tunnel",
    "댐": "dam",
    "탑": "tower",
    "교회": "church",
    "성당": "cathedral",
    "사원": "temple",
    "학교": "school",
    "병원": "hospital",
    "공장": "factory",

    # ── 전자기기 / IT
    "스마트폰": "smartphone",
    "핸드폰": "mobile phone",
    "폰": "phone",
    "스마트폰 화면": "phone screen",
    "노트북": "laptop",
    "컴퓨터": "computer",
    "모니터": "monitor",
    "키보드": "keyboard",
    "마우스": "computer mouse",
    "태블릿": "tablet",
    "TV": "television",
    "텔레비전": "television",
    "리모컨": "remote control",
    "카메라": "camera",
    "DSLR": "camera",
    "이어폰": "earphone",
    "헤드폰": "headphone",
    "스피커": "speaker",
    "프린터": "printer",
    "라우터": "router",

    # ── 가전제품
    "냉장고": "refrigerator",
    "세탁기": "washing machine",
    "에어컨": "air conditioner",
    "선풍기": "electric fan",
    "전자레인지": "microwave oven",
    "오븐": "oven",
    "청소기": "vacuum cleaner",
    "다리미": "iron",
    "헤어드라이어": "hair dryer",

    # ── 의류 / 패션
    "티셔츠": "t-shirt",
    "셔츠": "shirt",
    "바지": "pants",
    "청바지": "jeans",
    "치마": "skirt",
    "원피스": "dress",
    "재킷": "jacket",
    "코트": "coat",
    "패딩": "puffer jacket",
    "점퍼": "jacket",
    "운동화": "sneakers",
    "신발": "shoes",
    "구두": "dress shoes",
    "부츠": "boots",
    "슬리퍼": "slippers",
    "모자": "hat",
    "야구모자": "baseball cap",
    "니트": "knit sweater",
    "스웨터": "sweater",
    "후드티": "hoodie",
    "가방": "bag",
    "핸드백": "handbag",
    "배낭": "backpack",
    "지갑": "wallet",
    "벨트": "belt",
    "넥타이": "tie",
    "목도리": "scarf",
    "장갑": "glove",
    "양말": "socks",
    "선글라스": "sunglasses",
    "안경": "glasses",

    # ── 스포츠 / 운동 / 선수
    "축구공": "soccer ball",
    "농구공": "basketball",
    "야구공": "baseball",
    "배구공": "volleyball",
    "테니스공": "tennis ball",
    "골프채": "golf club",
    "라켓": "racket",
    "자전거 헬멧": "bicycle helmet",
    "스케이트보드": "skateboard",
    "서핑보드": "surfboard",
    "수영복": "swimsuit",
    "운동복": "sportswear",
    "트레드밀": "treadmill",
    "유니폼": "uniform",
    "축구선수": "soccer player",
    "야구선수": "baseball player",
    "농구선수": "basketball player",
    "배구선수": "volleyball player",
    "수영선수": "swimmer",
    "테니스선수": "tennis player",
    "골프선수": "golfer",
    "운동선수": "athlete",
    "선수": "athlete",
    "코치": "coach",
    "심판": "referee",
    "골키퍼": "goalkeeper",
    "축구팀": "soccer team",

    # ── 사람 / 역할
    "사람": "person",
    "남자": "man",
    "여자": "woman",
    "남성": "male",
    "여성": "female",
    "아이": "child",
    "어린이": "child",
    "아기": "baby",
    "노인": "elderly person",
    "할아버지": "old man",
    "할머니": "old woman",
    "학생": "student",
    "의사": "doctor",
    "간호사": "nurse",
    "경찰": "police officer",
    "소방관": "firefighter",
    "군인": "soldier",
    "요리사": "chef",
    "웨이터": "waiter",
    "운전사": "driver",
    "기사": "driver",
    "노동자": "worker",
    "작업자": "worker",

    # ── 자연 / 식물
    "나무": "tree",
    "꽃": "flower",
    "장미": "rose",
    "빨간 장미": "red rose",
    "해바라기": "sunflower",
    "튤립": "tulip",
    "벚꽃": "cherry blossom",
    "잎": "leaf",
    "나뭇잎": "tree leaf",
    "풀": "grass",
    "잔디": "lawn",
    "선인장": "cactus",
    "대나무": "bamboo",
    "버섯": "mushroom",
    "산": "mountain",
    "강": "river",
    "바다": "ocean",
    "호수": "lake",
    "하늘": "sky",
    "구름": "cloud",
    "눈": "snow",
    "비": "rain",
    "무지개": "rainbow",
    "바위": "rock",
    "모래": "sand",
    "흙": "soil",

    # ── 신체 부위
    "얼굴": "face",
    "눈": "eye",
    "코": "nose",
    "입": "mouth",
    "귀": "ear",
    "손": "hand",
    "발": "foot",
    "다리": "leg",
    "팔": "arm",
    "손가락": "finger",
    "발가락": "toe",
    "머리": "head",
    "어깨": "shoulder",
    "가슴": "chest",
    "배": "belly",
    "등": "back",
    "목": "neck",
    "머리카락": "hair",
    "입술": "lips",
    "눈썹": "eyebrow",

    # ── 생활 / 기타
    "박스": "box",
    "상자": "box",
    "팔레트": "pallet",
    "우산": "umbrella",
    "카트": "cart",
    "유모차": "stroller",
    "휠체어": "wheelchair",
    "쓰레기통": "trash can",
    "소화기": "fire extinguisher",
    "소화전": "fire hydrant",
    "전신주": "utility pole",
    "전봇대": "utility pole",
    "파이프": "pipe",
    "책": "book",
    "노트": "notebook",
    "연필": "pencil",
    "볼펜": "pen",
    "가위": "scissors",
    "칼": "knife",
    "포크": "fork",
    "숟가락": "spoon",
    "젓가락": "chopsticks",
    "열쇠": "key",
    "자물쇠": "lock",
    "동전": "coin",
    "지폐": "banknote",
    "카드": "card",
    "우편함": "mailbox",
    "종이": "paper",
    "봉투": "envelope",
    "테이프": "tape",
    "빗자루": "broom",
    "대걸레": "mop",
    "양동이": "bucket",
    "사다리": "ladder",
    "드릴": "drill",
    "망치": "hammer",
    "렌치": "wrench",
    "스패너": "spanner",
    "드라이버": "screwdriver",
}

_ENGLISH_PROMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '/,\-]*$")


## =====================================
## 함수 기능 : 입력 텍스트를 공백 정규화, 제어문자 제거해 반환합니다
## 매개 변수 : text(str)
## 반환 결과 : str -> 정규화된 텍스트
## =====================================
def normalize_user_text(text: str) -> str:
    normalized = str(text or "")
    normalized = normalized.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


## =====================================
## 함수 기능 : 텍스트가 SAM3에 유효한 1~4단어 영어 프롬프트인지 판별합니다
## 매개 변수 : text(str)
## 반환 결과 : bool -> 유효 여부
## =====================================
def _looks_like_english_prompt(text: str) -> bool:
    candidate = normalize_user_text(text)
    if not candidate:
        return False
    if not _ENGLISH_PROMPT_RE.fullmatch(candidate):
        return False
    words = [word for word in candidate.split() if word]
    return 1 <= len(words) <= 4


## =====================================
## 함수 기능 : 한국어 입력에서 공백을 제거한 붙여쓰기(compact) 형태를 반환합니다
##            예: "자동차 후미등" → "자동차후미등"
## 매개 변수 : text(str)
## 반환 결과 : str -> 공백 없는 텍스트
## =====================================
def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


## =====================================
## 함수 기능 : 한국어 입력 텍스트에서 객체 키워드를 찾아 스팬(start, end) 기반으로 반환합니다
##            - 2글자 이하 토큰은 공백으로 분리된 단어 단위에서만 매칭합니다 (오매칭 방지)
##            - 긴 매칭이 짧은 매칭을 포함하면 짧은 쪽을 제거합니다 (예: "강아지 귀"가 "귀"를 포함)
## 매개 변수 : text(str) -> 사용자 입력 텍스트
## 반환 결과 : list[tuple[str, str]] -> (영어 프롬프트, 매칭된 한국어 키) 목록 (긴 키 우선 정렬)
## =====================================
def _find_object_matches(text: str) -> list[tuple[str, str]]:
    lowered = text.casefold()
    word_set = set(lowered.split())  # 공백 분리 단어 집합 (짧은 토큰 경계 검사용)

    # (start, end, english, korean_key) 후보 수집
    candidates: list[tuple[int, int, str, str]] = []
    for token, mapped in _KOREAN_OBJECT_MAP.items():
        token_lower = token.casefold()

        # 1글자 토큰: 반드시 독립 단어로만 허용 (예: "파" → "green onion"이 "파란" 안에서 오매칭 방지)
        if len(token_lower) == 1 and token_lower not in word_set:
            continue

        pos = lowered.find(token_lower)
        if pos < 0:
            # 공백 제거 compact 폼으로 재시도 (예: "앞범퍼" ↔ "앞 범퍼")
            compact_text = _compact(lowered)
            compact_token = _compact(token_lower)
            if compact_token and compact_token in compact_text:
                pos = compact_text.find(compact_token)
                candidates.append((pos, pos + len(compact_token), mapped, token))
        else:
            candidates.append((pos, pos + len(token_lower), mapped, token))

    # 1순위: 스팬 길이 내림차순, 2순위: 시작 위치 오름차순
    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

    # 탐욕적(greedy) 겹침 제거: 긴 매칭이 먼저 선택되고, 그것과 겹치는 짧은 매칭은 제외
    selected: list[tuple[int, int, str, str]] = []
    for start, end, en, ko in candidates:
        overlaps = any(
            start < sel_end and end > sel_start
            for sel_start, sel_end, _, _ in selected
        )
        if not overlaps:
            selected.append((start, end, en, ko))

    # 텍스트 위치 오름차순 정렬: 한국어 수식어-주어 구조에 따라 앞 → 뒤 순서로 영어 복합어 생성
    selected.sort(key=lambda x: x[0])

    # 영어 중복 제거 후 반환
    seen_en: set[str] = set()
    result: list[tuple[str, str]] = []
    for _, _, en, ko in selected:
        en_key = en.casefold()
        if en_key not in seen_en:
            seen_en.add(en_key)
            result.append((en, ko))
    return result


## =====================================
## 함수 기능 : 한국어 입력에서 색상 키워드를 찾아 영어로 반환합니다
## 매개 변수 : text(str)
## 반환 결과 : str -> 매칭된 영어 색상, 없으면 빈 문자열
## =====================================
def _find_color_match(text: str) -> str:
    lowered = text.casefold()
    best = ("", 0)  # (영어색상, 키길이)
    for token, mapped in _KOREAN_COLOR_MAP.items():
        if token.casefold() in lowered and len(token) > best[1]:
            best = (mapped, len(token))
    return best[0]


## =====================================
## 함수 기능 : 여러 영어 토큰을 공백으로 합쳐 단어 수가 max_words 이하인 경우만 반환합니다
## 매개 변수 : parts(list[str]), max_words(int)
## 반환 결과 : str | None -> 결합된 구문 또는 None
## =====================================
def _join_if_short(parts: list[str], max_words: int = 4) -> str | None:
    phrase = " ".join(p.strip() for p in parts if p.strip())
    if len(phrase.split()) <= max_words:
        return phrase
    return None


## =====================================
## 함수 기능 : 한국어 입력 텍스트를 SAM3 영어 프롬프트 후보 목록으로 변환합니다
##            색상 + 수식 객체 + 주 객체를 결합해 "red uniform soccer player" 형태의
##            복합 구문을 포함한 후보를 생성합니다
## 매개 변수 : user_text(str), class_name(str)
## 반환 결과 : list[str] -> 영어 프롬프트 후보 목록
## =====================================
def heuristic_english_candidates(user_text: str, class_name: str = "") -> list[str]:
    normalized = normalize_user_text(user_text)
    normalized_class = normalize_user_text(class_name)
    candidates: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        value = normalize_user_text(text)
        if not value:
            return
        key = value.casefold()
        if key in seen:
            return
        seen.add(key)
        candidates.append(value)

    color_word = _find_color_match(normalized)
    object_matches = _find_object_matches(normalized)

    if object_matches:
        # object_matches는 텍스트 위치 오름차순 (한국어: 수식어 → 주 객체 순서)
        # 마지막 매칭이 의미상 핵심 객체(head), 앞의 것들이 수식어(modifier)
        all_en = [en for en, _ in object_matches]
        head_en = all_en[-1]  # 주 객체 (텍스트에서 가장 마지막)

        # ── 복합 구문: [색상] + en1 + en2 + ... (텍스트 위치 순서 그대로)
        # 예: "자동차 후미등" → "car tail light", "빨간 유니폼 축구선수" → "red uniform soccer player"
        if len(all_en) >= 2:
            # 색상이 이미 첫 번째 객체에 포함된 경우 중복 방지
            first_already_has_color = bool(
                color_word and all_en[0].casefold().startswith(color_word.casefold())
            )
            if color_word and not first_already_has_color:
                compound = _join_if_short([color_word] + all_en)
                if compound:
                    add(compound)
            compound_no_color = _join_if_short(all_en)
            if compound_no_color:
                add(compound_no_color)

        # ── 색상 + 주 객체 (단독)
        # 주 객체에 이미 색상이 포함된 경우 중복 방지
        head_already_has_color = bool(
            color_word and head_en.casefold().startswith(color_word.casefold())
        )
        if color_word and not head_already_has_color:
            add(f"{color_word} {head_en}")

        # ── 주 객체 단독
        add(head_en)

        # ── 수식 객체들 (색상 조합 포함)
        for en in all_en[:-1]:
            already_has_color = bool(
                color_word and en.casefold().startswith(color_word.casefold())
            )
            if color_word and not already_has_color:
                add(f"{color_word} {en}")
            add(en)

    elif color_word:
        # 객체 매칭 없고 색상만 있는 경우
        if normalized_class and _looks_like_english_prompt(normalized_class):
            add(f"{color_word} {normalized_class}")
        add(color_word)

    # class_name이 영어 프롬프트이면 추가
    if normalized_class and _looks_like_english_prompt(normalized_class):
        add(normalized_class)

    return candidates


## =====================================
## 함수 기능 : LLM 랭킹 결과 payload에서 영어 프롬프트 후보 목록을 추출합니다
## 매개 변수 : payload(Mapping|None)
## 반환 결과 : list[str] -> 영어 프롬프트 목록
## =====================================
def extract_ranked_prompt_candidates(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if not isinstance(items, Sequence):
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        prompt = normalize_user_text(str(item.get("english_prompt", "")))
        if not _looks_like_english_prompt(prompt):
            continue
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(prompt)
    return candidates


## =====================================
## 함수 기능 : 모든 소스(LLM 결과, 휴리스틱)를 합쳐 최종 SAM3 프롬프트 후보 목록을 반환합니다
## 매개 변수 : prompt_text(str), class_name(str), ranked_candidates(list|None), limit(int)
## 반환 결과 : list[str] -> SAM3에 전달할 영어 프롬프트 후보 목록
## =====================================
def build_sam_prompt_candidates(
    *,
    prompt_text: str,
    class_name: str = "",
    ranked_candidates: Sequence[str] | None = None,
    limit: int = 8,
) -> list[str]:
    base_prompt = normalize_user_text(prompt_text)
    base_class = normalize_user_text(class_name)
    variants: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        value = normalize_user_text(raw)
        if not value:
            return
        key = value.casefold()
        if key in seen:
            return
        seen.add(key)
        variants.append(value)

    def add_with_period(raw: str) -> None:
        value = normalize_user_text(raw)
        if not value:
            return
        add(value)
        if not value.endswith("."):
            add(f"{value}.")

    heuristic_candidates = list(heuristic_english_candidates(base_prompt, base_class))

    if _looks_like_english_prompt(base_prompt):
        # 영어 입력이면 그대로 우선 사용
        add_with_period(base_prompt)
        prompt_tokens = [token.strip(" .,_-/") for token in base_prompt.split() if token.strip(" .,_-/")]
        if prompt_tokens:
            add_with_period(prompt_tokens[-1])
        if len(prompt_tokens) >= 2:
            add_with_period(" ".join(prompt_tokens[-2:]))
        for candidate in ranked_candidates or []:
            candidate_text = normalize_user_text(str(candidate))
            if _looks_like_english_prompt(candidate_text):
                add_with_period(candidate_text)
        for candidate in heuristic_candidates:
            add_with_period(candidate)
    else:
        # 한국어 입력: 휴리스틱(구체적 부품명) 우선, 그 다음 LLM 결과
        for candidate in heuristic_candidates:
            add_with_period(candidate)
        for candidate in ranked_candidates or []:
            candidate_text = normalize_user_text(str(candidate))
            if _looks_like_english_prompt(candidate_text):
                add_with_period(candidate_text)

    if _looks_like_english_prompt(base_class):
        add_with_period(base_class)
        class_tokens = [token.strip(" .,_-/") for token in base_class.split() if token.strip(" .,_-/")]
        if class_tokens:
            add_with_period(class_tokens[-1])

    if not variants:
        add_with_period("object")
    return variants[: max(1, int(limit))]


## =====================================
## 함수 기능 : 휴리스틱 후보를 우선으로 하고 LLM 결과(한글 주석 포함)로 보완한
##            UI 표시 전용 payload를 생성합니다.
##            - 휴리스틱 결과가 먼저 나열되고, 거기 없는 LLM 항목이 뒤에 추가됩니다.
##            - "uniform soccer player athlete" 같은 LLM 오생성 결과가 있어도
##              휴리스틱이 앞에 나오므로 사용자에게는 올바른 순서로 표시됩니다.
## 매개 변수 : user_text(str), class_name(str), llm_payload(Mapping|None), limit(int)
## 반환 결과 : dict -> format_nlp_output_for_display 에 전달할 payload
## =====================================
def build_display_payload(
    *,
    user_text: str,
    class_name: str = "",
    llm_payload: Mapping[str, Any] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    heuristic = heuristic_english_candidates(user_text, class_name)

    # LLM payload에서 (english_prompt, korean_gloss) 추출
    llm_items: list[tuple[str, str]] = []
    if isinstance(llm_payload, Mapping):
        raw_items = llm_payload.get("items") or []
        if isinstance(raw_items, Sequence):
            for item in raw_items:
                if not isinstance(item, Mapping):
                    continue
                en = normalize_user_text(str(item.get("english_prompt", "")))
                ko = normalize_user_text(str(item.get("korean_gloss", "")))
                if en and _looks_like_english_prompt(en):
                    llm_items.append((en, ko))

    # LLM 한글 주석 조회용 딕셔너리
    llm_ko_map: dict[str, str] = {en.casefold(): ko for en, ko in llm_items if ko}

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    def _add(en: str, ko: str) -> None:
        key = en.casefold()
        if key in seen or len(merged) >= limit:
            return
        seen.add(key)
        merged.append({"english_prompt": en, "korean_gloss": ko})

    # 1순위: 휴리스틱 후보 (LLM 한글 주석이 있으면 함께 표시)
    for en in heuristic:
        ko = llm_ko_map.get(en.casefold(), "")
        _add(en, ko)

    # 2순위: 휴리스틱에 없는 LLM 항목 (보완)
    for en, ko in llm_items:
        _add(en, ko)

    return {
        "model_id": (llm_payload or {}).get("model_id", "heuristic"),
        "items": merged,
        "_meta": dict((llm_payload or {}).get("_meta") or {}),
    }


## =====================================
## 함수 기능 : LLM 결과 payload를 UI 표시용 "추천 클래스 목록" 문자열로 포매팅합니다
## 매개 변수 : payload(Mapping|None)
## 반환 결과 : str -> 표시용 텍스트
## =====================================
def format_nlp_output_for_display(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    items = payload.get("items")
    if not isinstance(items, Sequence) or not items:
        return ""
    lines = ["추천 클래스 목록"]
    count = 0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        en = normalize_user_text(str(item.get("english_prompt", "")))
        ko = normalize_user_text(str(item.get("korean_gloss", "")))
        if not en:
            continue
        count += 1
        if ko and re.search(r"[\uac00-\ud7a3]", ko):
            lines.append(f"{count}. {en} ({ko})")
        else:
            lines.append(f"{count}. {en}")
    return "\n".join(lines) if count > 0 else ""


## =====================================
## 함수 기능 : LLM 랭킹 결과를 UI 목록에 표시할 라인 목록으로 포매팅합니다 (확률 표시 없음)
## 매개 변수 : payload(Mapping|None)
## 반환 결과 : list[str] -> "영어 | 한글" 형식의 라인 목록
## =====================================
def format_ranked_prompt_lines(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    lines: list[str] = []
    items = payload.get("items")
    if not isinstance(items, Sequence):
        return lines
    for item in items:
        if not isinstance(item, Mapping):
            continue
        en = normalize_user_text(str(item.get("english_prompt", "")))
        ko = normalize_user_text(str(item.get("korean_gloss", "")))
        if ko and not re.search(r"[\uac00-\ud7a3]", ko):
            ko = ""
        if not en:
            continue
        if ko:
            lines.append(f"{en} | {ko}")
        else:
            lines.append(en)
    return lines
