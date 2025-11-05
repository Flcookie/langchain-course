from ipaddress import summarize_address_range
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    黄仁勋（Jensen Huang），美籍华人，1963年2月17日出生于台湾省台南市，美国工程院院士 [53]。NVIDIA公司创始人兼首席执行官 [12] [14]。
1983年，黄仁勋毕业于俄勒冈州立大学，1990年取得斯坦福大学电子工程硕士学位，先后在AMD、LSI Logic两家公司任职芯片工程师等工作。1993年，黄仁勋创立NVIDIA公司，公司在1999年发明GPU，让实时可编程着色技术成为可能，这一技术也定义了现代计算机图形及后来革命性的并行计算。2006年，推出并行计算平台和编程模型“CUDA”，通过GPU实现多个领域的高效计算，为后来的人工智能发展提供了动力 [12] [14] [42]。
2001年，黄仁勋以5.07亿美元的身价名列《财富》杂志评选“40岁以下的40位富翁”榜单第12位。此外，他曾获美国半导体行业协会（SIA）最高荣誉罗伯特·诺伊斯奖、IEEE 创始人奖章、张忠谋博士模范领袖奖等多个奖项荣誉。2021年11月，入选“福布斯中国·北美华人精英TOP60”。2023年9月8日，被评为全球AI领袖 [5] [10] [12] [14]。2024年，当选美国工程院院士 [53]。同年，以3500亿财富名列“2024胡润全球富豪榜”，位列榜单26位 [55]。截至2024年5月30日，黄仁勋的资产净值首次突破1000亿美元，位列全球富豪榜第15 [59]。2024年10月7日，黄仁勋个人身价已达1090亿美元，排名全球富豪榜第13名。 [62]2025年3月27日，胡润研究院发布《2025胡润全球富豪榜》，黄仁勋以9350亿元人民币财富位列榜单第11位 [72]。10月29日，英伟达股价大幅上涨，黄仁勋位列福布斯富豪榜第八位 [122]
    """
    summary_template = f"""
    given the information {information} about a person I want you to create:
    1.A short summary 
    2.two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    #llm = ChatOllama(temperature=0, model = "gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
