# Get a new token: https://help.aliyun.com/document_detail/611472.html?spm=a2c4g.2399481.0.0
from getpass import getpass
from langchain_community.llms import Tongyi
import os
import dotenv

# DASHSCOPE_API_KEY = getpass()
# os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
# dotenv.load_dotenv()
# os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
# llm = Tongyi(model="qwen-plus")
# print(llm.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?"))


import os
from langchain_community.llms import Tongyi
# os.environ["DASHSCOPE_API_KEY"] = 'sk-0a5edb702c6d41a89ca289c0d2363843'
llm = Tongyi(model="qwen-plus", temperature=0.1)
print(llm.invoke("你是谁？"))





