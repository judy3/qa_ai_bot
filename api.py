import logging

import tornado.ioloop
import tornado.web

from bot_utils import ask_bot, load_embedding_model, load_rwkv_model, ask_bot_without_kb

embeddings = load_embedding_model()
rwkv_model = load_rwkv_model()

class CloneHandler(tornado.web.RequestHandler):
    def post(self):
        #embeddings = load_embedding_model()
        #rwkv_model = load_rwkv_model()
        data = tornado.escape.json_decode(self.request.body)
        question = data.get("question")

        response = ask_bot(
            embeddings=embeddings,
            rwkv_model=rwkv_model,
            question=question
        )
        self.write(response)

class AskBotWithoutKbHandler(tornado.web.RequestHandler):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        question = data.get("question")

        response = ask_bot_without_kb(
            rwkv_model=rwkv_model,
            question=question
            )
        self.write(response)


app = tornado.web.Application()
app.add_handlers(r".*", [
    (r"/ask_bot", CloneHandler),            # 第一个URI路径
    (r"/ask_bot_without_kb", AskBotWithoutKbHandler)  # 第二个URI路径
    ])


if __name__ == "__main__":
    logging.info("Starting the ask_bot api...")
    app.listen(8080)
    tornado.ioloop.IOLoop.current().start()
