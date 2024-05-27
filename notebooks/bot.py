import asyncio
import pickle
import string

from aiogram.types import Message
from aiogram.filters import CommandStart

from aiogram import Bot, Dispatcher
from nltk import SnowballStemmer
from nltk.corpus import stopwords

bot = Bot(token='6918378329:AAGJOGVrK0PGiLfcGQsKCfzJwYOSiKk8Xe8')
dp = Dispatcher()

stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

with open('model.pkl', 'rb') as f:
    vectorizer, mdl = pickle.load(f)


def prepare(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word.lower() not in stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


def predict(data):
    prepared_text = prepare(data)
    x = vectorizer.transform([prepared_text])
    return mdl.predict(x)[0]


@dp.message(CommandStart())
async def hello(message: Message):
    await message.answer("Hello!")


@dp.message()
async def echo(message: Message):
    res = predict(str(message.text))
    await message.reply(str(res))


async def start():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == '__main__':
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print('Bot is off')
