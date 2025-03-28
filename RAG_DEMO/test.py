# Databricks notebook source
# MAGIC %pip install -U -qqqq pypdf unstructured databricks-agents mlflow mlflow-skinny databricks-vectorsearch databricks-sdk langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet databricks-sdk langchain-community databricks-langchain langgraph mlflow googlemaps
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION hhhd_demo_itec.default.magic_function(input INT)
# MAGIC RETURNS INT
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC   return input + 2
# MAGIC $$;
# MAGIC

# COMMAND ----------

import googlemaps

def find_nearest_station(address: str) -> str:
    api_key = "AIzaSyDU8HiLkRbwzqG2kCljLiUvD1TyuTi6JpQ"
    gmaps = googlemaps.Client(key=api_key)

    geocode_result = gmaps.geocode(address)
    if not geocode_result:
        return "住所が見つかりませんでした"

    location = geocode_result[0]['geometry']['location']
    lat, lng = location['lat'], location['lng']

    places = gmaps.places_nearby(location=(lat, lng), radius=1000, type='train_station')
    if places.get("results"):
        return places['results'][0]['name']

    return "最寄り駅が見つかりませんでした"

result = find_nearest_station("大阪府大阪市福島区大開１丁目８−１")
print(result)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP FUNCTION IF EXISTS hhhd_demo_itec.default.find_nearest_station;
# MAGIC
# MAGIC CREATE FUNCTION hhhd_demo_itec.default.find_nearest_station(address STRING)
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC AS $$
# MAGIC import googlemaps
# MAGIC
# MAGIC api_key = "AIzaSyDU8HiLkRbwzqG2kCljLiUvD1TyuTi6JpQ"
# MAGIC gmaps = googlemaps.Client(key=api_key)
# MAGIC
# MAGIC geocode_result = gmaps.geocode(address)
# MAGIC if not geocode_result:
# MAGIC   return "住所が見つかりませんでした"
# MAGIC
# MAGIC location = geocode_result[0]['geometry']['location']
# MAGIC lat, lng = location['lat'], location['lng']
# MAGIC
# MAGIC places = gmaps.places_nearby(location=(lat, lng), radius=1000, type='train_station')
# MAGIC if places.get("results"):
# MAGIC   return places['results'][0]['name']
# MAGIC
# MAGIC return "最寄り駅が見つかりませんでした"
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE FUNCTION hhhd_demo_itec.default.find_nearest_station;

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from databricks.sdk import WorkspaceClient
from langchain_community.tools.databricks import UCFunctionToolkit

# Databricksのエンドポイントに接続
llm = ChatDatabricks(
  endpoint="databricks-meta-llama-3-3-70b-instruct",
  extra_params={"temperature": 0.01},
)

tools = (
    UCFunctionToolkit(
        # SQLウェアハウスID
        warehouse_id="7bcc7ca11b4aa98c"
    )
    .include(
        # "{catalog_name}.{schema_name}.*" を使用してスキーマ内のすべての関数を取得できます。
        "hhhd_demo_itec.default.*",
    )
    .get_tools()
)

import os

os.environ["UC_TOOL_CLIENT_EXECUTION_TIMEOUT"] = "200"

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは役に立つアシスタントです。ツールを使用してください。",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "3"})


# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from databricks.sdk import WorkspaceClient
from langchain_community.tools.databricks import UCFunctionToolkit

# Databricksのエンドポイントに接続
model = ChatDatabricks(
  endpoint="databricks-meta-llama-3-3-70b-instruct",
  extra_params={"temperature": 0.01},
)

tools = (
    UCFunctionToolkit(
        # SQLウェアハウスID
        warehouse_id="7bcc7ca11b4aa98c"
    )
    .include(
        # "{catalog_name}.{schema_name}.*" を使用してスキーマ内のすべての関数を取得できます。
        "hhhd_demo_itec.default.*",
    )
    .get_tools()
)

import os
os.environ["UC_TOOL_CLIENT_EXECUTION_TIMEOUT"] = "200"

station_check_instruction = """
    あなたは通勤手当申請のチェックを行うアシスタントです。

    申請書データから、各申請者の「自宅住所」を使って、最寄り駅を検索してください。
    その上で、「申請書に記載された最寄り駅」と一致するかどうかを確認し、結果を出力してください。

    注意：
    - 複数の申請者が含まれている場合は、申請者ごとに処理してください。
    - 違う申請者の判断結果を参照しないでください。
    - 出力フォーマットは以下の通りです（この形式に厳密に従ってください）：

    申請者名: 
    申請書に記載された最寄り駅:
    検索された最寄り駅:
    判断結果:
    """

# ChatPromptTemplate作成
station_check_prompt = ChatPromptTemplate.from_messages([
  ("system", station_check_instruction),
  ("placeholder", "{chat_history}"),
  ("user", "{input_text}"),
  ("placeholder", "{agent_scratchpad}")
])

# エージェントと実行器作成
agent = create_tool_calling_agent(model, tools, station_check_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def check_station_match_with_llm(text: str) -> str:
  input_text = f"申請書データ: {text}"
  result = agent_executor.invoke({"input_text": input_text})
  return result["output"] if isinstance(result, dict) and "output" in result else str(result)

# input_example = """
#                 申請者名: 田中 太郎 自宅住所: 大阪府大阪市浪速区敷津東１丁目４−２０
#                 申請者名: 大阪 太郎 自宅住所: 大阪府大阪市福島区大開１丁目８−１ 
#             """

input_example = """
申請者名: 田中 太郎
自宅住所: 大阪府大阪市浪速区敷津東１丁目４−２０

申請者名: 大阪 太郎
自宅住所: 大阪府大阪市福島区大開１丁目８−１
"""

check_station_match_with_llm(input_example)

# COMMAND ----------

# @tool
# def find_nearest_station(address: str) -> str:
#     """住所から最寄りの駅を返します"""
#     api_key = "AIzaSyDU8HiLkRbwzqG2kCljLiUvD1TyuTi6JpQ"
#     gmaps = googlemaps.Client(key=api_key)

#     geocode_result = gmaps.geocode(address)
#     if not geocode_result:
#         return "住所が見つかりませんでした"

#     location = geocode_result[0]['geometry']['location']
#     lat, lng = location['lat'], location['lng']

#     places = gmaps.places_nearby(location=(lat, lng), radius=1000, type='train_station')
#     if places.get("results"):
#         return places['results'][0]['name']

#     return "最寄り駅が見つかりませんでした"
