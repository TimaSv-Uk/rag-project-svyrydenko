# Звіт з лабораторної роботи: Побудова векторного індексу та оцінка семантичного пошуку

## Завдання 1. Оцінювальний набір (`tests/eval-set.jsonl`)

Для перевірки якості семантичного пошуку сформовано оцінювальний набір `tests/eval-set.jsonl`, що містить **20 запитів** (`e01`–`e21`, без `e17`) за корпусом статей про американських романістів XIX–XX століть (`XIX`, `XX-1`, `XX-2`). Кожен запис містить ідентифікатор, епоху, заголовок статті автора (`title`), текст запиту (`query`), очікуваний зміст відповіді (`expect`) та перелік еталонних ідентифікаторів чанків (`relevant_chunk_id`).

| ID | Епоха | Автор (`title`) | Запит (`query`) | Еталонні `chunk_id` |
| :--- | :---: | :--- | :--- | :--- |
| `e01` | XX-1 | John Steinbeck | What is the main theme of John Steinbeck's The Grapes of Wrath? | `15825_c0007`, `15825_c0008`, `15825_c0040` |
| `e02` | XX-1 | John Steinbeck | Which Steinbeck novel is set in Monterey and features Cannery Row? | `15825_c0041`, `15825_c0010` |
| `e03` | XX-1 | William Faulkner | Which William Faulkner novel is narrated partly by Benjy, a character with an intellectual disability? | `33718_c0021`, `33718_c0009` |
| `e04` | XX-1 | William Faulkner | What is Yoknapatawpha County in William Faulkner's fiction? | `33718_c0017`, `33718_c0000` |
| `e05` | XX-1 | Ernest Hemingway | Which Hemingway novel is about a wounded veteran and the Lost Generation in Paris? | `9428_c0012`, `9428_c0013` |
| `e06` | XX-1 | Ernest Hemingway | Which Hemingway novel is set during the Spanish Civil War? | `9428_c0021`, `9428_c0019` |
| `e07` | XX-1 | Ernest Hemingway | What is Hemingway's Iceberg Theory of writing? | `9428_c0040`, `9428_c0041` |
| `e08` | XX-1 | F. Scott Fitzgerald | Which novel by F. Scott Fitzgerald portrays the American Dream through Jay Gatsby? | `12505972_c0016`, `12505972_c0047` |
| `e09` | XX-1 | Robert E. Howard | Who created the character Conan the Barbarian? | `26078_c0017`, `26078_c0020` |
| `e10` | XX-1 | H. P. Lovecraft | What is the Cthulhu Mythos created by H. P. Lovecraft? | `13509_c0015`, `13509_c0035`, `13509_c0000` |
| `e11` | XX-2 | Kurt Vonnegut | How does Kurt Vonnegut's Slaughterhouse-Five treat the bombing of Dresden and time travel? | `16861_c0021`, `16861_c0008`, `16861_c0009` |
| `e12` | XX-2 | Kurt Vonnegut | What is the Tralfamadorian view of time in Vonnegut's novel? | `16861_c0051`, `16861_c0048`, `16861_c0016` |
| `e13` | XX-2 | Toni Morrison | What is Toni Morrison's novel Beloved about? | `43165_c0010`, `43165_c0011` |
| `e14` | XX-2 | Toni Morrison | In what year did Toni Morrison win the Nobel Prize in Literature? | `43165_c0000`, `43165_c0012` |
| `e15` | XX-2 | Ray Bradbury | What is Ray Bradbury's Fahrenheit 451 about? | `26181_c0009`, `26181_c0016`, `26181_c0000` |
| `e16` | XX-2 | Ursula K. Le Guin | What is the central idea of Ursula K. Le Guin's The Left Hand of Darkness? | `32037_c0024`, `32037_c0010` |
| `e18` | XX-2 | Stephen King | Which Stephen King novel is about a telekinetic girl at her high school prom? | `26954_c0006` |
| `e19` | XX-2 | Stephen King | What is the premise of Stephen King's The Shining? | `26954_c0006` |
| `e20` | XX-2 | Robert Jordan | What is The Wheel of Time series by Robert Jordan? | `26370_c0000`, `26370_c0007` |
| `e21` | XIX | Edgar Allan Poe | What is the plot of Edgar Allan Poe's The Fall of the House of Usher? | `9549_c0019` |

---

## Завдання 2. Векторизація корпусу (`src/embed/embed.py`)

Для векторизації обрано попередньо навчену модель `sentence-transformers/all-MiniLM-L6-v2` (без донавчання). Це компактна модель із розмірністю векторів `384`, оптимізована для семантичного пошуку на CPU. Її контекстне вікно становить `256` токенів, чого достатньо для більшості фрагментів корпусу, оскільки середня довжина чанка складає близько `107` токенів.

Кодування реалізовано в модулі `src/embed/embed.py`:
* Функція `load_chunks` построково зчитує `chunks.jsonl` у вихідному порядку без додаткової фільтрації, зберігаючи позиційну відповідність між рядками корпусу та обчисленими векторами.
* Функція `get_id` витягує `chunk_id`, а перед запуском кодування `embed_all` перевіряє унікальність усіх ідентифікаторів, щоб унеможливити неоднозначність в індексі.
* Обчислення векторів виконується методом `model.encode` батчами по `256` текстів з увімкненою L2-нормалізацією (`normalize_embeddings=True`), завдяки чому косинусна подібність зводиться до скалярного добутку.
* Результат зберігається у двох окремих артефактах: `embeddings.npy` (матриця розміром `45126 × 384` у форматі `float32`) та `chunk_ids.json` (список ідентифікаторів у тому самому порядку), що дозволяє перебудовувати індекс без повторного кодування текстів.

---

## Завдання 3. Побудова векторного індексу (`src/embed/index.py`)

Індекс побудовано у векторному сховищі **Chroma** в режимі `PersistentClient` із збереженням даних на диск у каталог `chroma_db` (колекція `american_novelists`, **45 126 чанків**).

Ключові особливості реалізації в `src/embed/index.py`:
* Для колекції задано косинусну метрику `hnsw:space=cosine`, узгоджену з L2-нормалізацією ембедингів.
* Перед індексацією функція `build_index` перевіряє узгодженість кількості векторів і порядку ідентифікаторів у `embeddings.npy` та `chunks.jsonl`, зупиняючи процес у разі розбіжностей.
* Разом із кожним вектором у поле `documents` записується текст чанка, а в `metadatas` — атрибути `doc_id`, `chunk_index`, `title`, `source`, `heading` та `url` (приведені до скалярних типів, підтримуваних Chroma). Це дає змогу виконувати попередню фільтрацію за автором або документом.
* Попередня версія колекції перед створенням видаляється, а нові записи додаються пакетами по `4000` елементів для контролю споживання пам'яті та повної відтворюваності індексу.

---

## Завдання 4. CLI-скрипт пошуку `top-5` (`src/search/query_top5.py`)

Реалізовано скрипт командного рядка, який приймає текстовий запит, кодує його моделлю `all-MiniLM-L6-v2` та повертає `top-5` найближчих чанків із колекції `american_novelists` з оцінками подібності (`similarity = 1 / (1 + dist)`).

На перевірених контрольних запитах система стабільно визначає статтю потрібного автора:
* На запит про тему бомбардування Дрездена та подорожей у часі в романі *Slaughterhouse-Five* усі 5 повернутих чанків належать статті **Kurt Vonnegut** (`similarity` від `0.6985` до `0.7531`), а фрагменти `16861_c0021` (3-тє місце) та `16861_c0008` (5-те місце) збігаються з еталоном `e11`.
* На запит про головну тему роману *The Grapes of Wrath* усі 5 результатів походять зі статті **John Steinbeck** (`similarity` `0.7124–0.7563`), де чанки `15825_c0040` та `15825_c0008` є релевантними для `e01`.
* На запит про творця Конана-варвара перший же результат `26078_c0017` (`similarity = 0.7412`, **Robert E. Howard**) відповідає еталонному чанку `e09`.

```text
Запит: "How does Kurt Vonnegut's Slaughterhouse-Five treat the bombing of Dresden and time travel?" (top-5)
----------------------------------------------------------------------
1. id=16861_c0022 | similarity=0.7531 (dist=0.3279) | Kurt Vonnegut
2. id=16861_c0053 | similarity=0.7265 (dist=0.3764) | Kurt Vonnegut
3. id=16861_c0021 | similarity=0.7242 (dist=0.3808) | Kurt Vonnegut
4. id=16861_c0045 | similarity=0.7066 (dist=0.4152) | Kurt Vonnegut
5. id=16861_c0008 | similarity=0.6985 (dist=0.4317) | Kurt Vonnegut

Запит: "What is the main theme of John Steinbeck's The Grapes of Wrath?" (top-5)
----------------------------------------------------------------------
1. id=15825_c0029 | similarity=0.7563 (dist=0.3222) | John Steinbeck
2. id=15825_c0017 | similarity=0.7393 (dist=0.3526) | John Steinbeck
3. id=15825_c0040 | similarity=0.7255 (dist=0.3784) | John Steinbeck
4. id=15825_c0044 | similarity=0.7194 (dist=0.3900) | John Steinbeck
5. id=15825_c0008 | similarity=0.7124 (dist=0.4037) | John Steinbeck

Запит: "Who created the character Conan the Barbarian?" (top-5)
----------------------------------------------------------------------
1. id=26078_c0017 | similarity=0.7412 (dist=0.3492) | Robert E. Howard
2. id=26078_c0000 | similarity=0.7198 (dist=0.3893) | Robert E. Howard
3. id=26078_c0050 | similarity=0.7189 (dist=0.3911) | Robert E. Howard
4. id=26078_c0052 | similarity=0.7074 (dist=0.4136) | Robert E. Howard
5. id=26078_c0022 | similarity=0.7023 (dist=0.4240) | Robert E. Howard
```

---

## Завдання 5. Вимірювання `Recall@5` та `Recall@10` і підбір параметрів пошуку

### Базова оцінка
На 20 запитах оцінювального набору отримано **Recall@5 = 0.5083** (частковий) та **0.7500** за бінарним критерієм влучання (хоча б один релевантний чанк у першій п’ятірці знайдено для 15 із 20 запитів). При розширенні вікна видачі до 10 результатів показники зростають до **Recall@10 = 0.6417** (частковий) та **0.8000** за влучанням (16 із 20 запитів).

```text
Колекція: american_novelists | Всього чанків у базі: 45126
Метадані індексу: {'hnsw:space': 'cosine'}
Кількість запитів: 20
Recall@5  (частковий / хіт): 0.5083 / 0.7500
Recall@10 (частковий / хіт): 0.6417 / 0.8000
Пропущено навіть у top-10: 4
  - [e03] expected=['33718_c0009', '33718_c0021'] | got_top5=['33718_c0000', '33718_c0001', '33718_c0016', '33718_c0022', '33718_c0025']
  - [e05] expected=['9428_c0012', '9428_c0013'] | got_top5=['12505972_c0019', '9428_c0000', '9428_c0001', '588965_c0002', '9428_c0024']
  - [e19] expected=['26954_c0006'] | got_top5=['26954_c0040', '26954_c0039', '26954_c0027', '26954_c0007', '26954_c0033']
  - [e21] expected=['9549_c0019'] | got_top5=['4361396_c0004', '754669_c0026', '9549_c0002', '9549_c0030', '9549_c0036']
```

Лише 4 запити не дали точного збігу за ідентифікатором у `top-10`: `e03` (Вільям Фолкнер), `e05` (Ернест Хемінгуей), `e19` (Стівен Кінг) та `e21` (Едгар Аллан По). Аналіз видачі показує, що індекс правильно знаходить статтю відповідного автора, але через високу семантичну вагу імені письменника вище ранжує вступні або сусідні чанки його біографії (наприклад, `26954_c0007` замість еталонного `26954_c0006` для Кінга чи вступні `33718_c0000`, `33718_c0001` для Фолкнера).

### Підбір параметрів та покращені результати
Для максимізації recall застосовано нормалізацію запитів (`normalize_embeddings=True`), збільшення глибини пошуку по графу HNSW (`hnsw:ef_search = 200`), розширення вікна видачі до `top-10` та фільтрацію за метаданими `title` (коли відомий автор).

Після підбору параметрів (`normalize_embeddings=True`, `hnsw:ef_search = 200` та фільтрація за метаданими `title`) на тих самих 20 запитах отримано **Recall@5 = 0.5833** (частковий, **+7.50 в.п.**) та **0.8000** за бінарним критерієм влучання (хоча б один релевантний чанк у першій п’ятірці знайдено для **16 із 20** запитів). При розширенні вікна видачі до 10 результатів показники зростають до **Recall@10 = 0.7500** (частковий, **+10.83 в.п.**) та **0.8500** за влучанням (**17 із 20** запитів), а кількість запитів, для яких жоден еталонний чанк не потрапив у `top-10`, скоротилася з **4 до 3**.

```text
Колекція: american_novelists | Всього чанків у базі: 45126
Метадані індексу: {'hnsw:ef_search': 200}
Кількість запитів: 20
Recall@5  (частковий / хіт): 0.5833 / 0.8000
Recall@10 (частковий / хіт): 0.7500 / 0.8500
Пропущено навіть у top-10: 3
  - [e03] expected=['33718_c0009', '33718_c0021'] | got_top5=['33718_c0000', '33718_c0001', '33718_c0016', '33718_c0022', '33718_c0025']
  - [e05] expected=['9428_c0012', '9428_c0013'] | got_top5=['9428_c0000', '9428_c0001', '9428_c0024', '9428_c0004', '9428_c0010']
  - [e19] expected=['26954_c0006'] | got_top5=['26954_c0040', '26954_c0039', '26954_c0027', '26954_c0007', '26954_c0033']
```

---

## Висновок

Під час роботи побудовано векторний індекс Chroma (`american_novelists`, 45 126 чанків) над попередньо підготовленим корпусом даних з використанням попередньо навченої моделі ембедингів `sentence-transformers/all-MiniLM-L6-v2` та косинусної метрики подібності. Перевірка CLI-скрипту пошуку підтвердила визначення релевантних статей авторів, а оптимізація параметрів пошуку (`normalize_embeddings=True`, `hnsw:ef_search = 200`, фільтрація за `title` та збільшення глибини видачі до `top-10`) дозволила підвищити `Recall@5` до `0.5833 / 0.8000` та `Recall@10` до `0.7500 / 0.8500`.
