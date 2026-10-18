# American Literature RAG

1. Опис проєкту

American Literature RAG — система пошуку та генерації відповідей на запитання про класичну американську літературу на основі статей Wikipedia.

2. Набір даних

Джерело: Wikipedia на Hugging Face Datasets.
Версія: 20231101.en — англомовна Wikipedia.
Вибірка: статті про американських романістів 19 та 20 століття, відібрані за категоріями Wikipedia.
Ліцензія: CC BY-SA 4.0 та GFDL відповідно до умов поширення матеріалів Wikipedia. Під час повторного використання необхідно дотримуватися умов атрибуції та відповідної ліцензії.

3. Технологічний стек

- Векторне сховище — Chroma
- Модель ембедингів — all-MiniLM-L6-v2
- Модель генерації — Qwen 2.5 7B
- Мова програмування — Python


4. Вимоги до системи

- Достатньо вільної оперативної пам'яті 8–16 ГБ RAM.

Перед встановленням підготуй:
- uv для підтримки віртуального середовища.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Ollama для локального запуску мовної моделі.
```
curl -fsSL https://ollama.com/install.sh | sh
```
- Python 3.11 або сумісну версію Python 3.10+.
```
uv python install --default
```

5. Встановлення
Наведені нижче команди призначені для Linux
```
git clone https://github.com/TimaSv-Uk/rag-project-svyrydenko
cd rag-project-svyrydenko
uv sync
ollama pull qwen2.5:7b
```
