import os

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
try:
    from dotenv import load_dotenv
except ImportError:
    raise RuntimeError('Python environment for SPARK AI is not completely set up: required package "python-dotenv" is missing.') from None

load_dotenv()


def initial_prompt_generate(source_lang: str, target_lang: str, source_text: str) -> str:
    prompt = f"""
    你是一位语言学专家，专门从事从{source_lang}到{target_lang}的翻译工作。
    下面是一个从{source_lang}到{target_lang}的翻译任务，请为文本{source_text}提供到{target_lang}的翻译。
    不要提供除翻译之外的任何解释或文字。
    {source_lang}: {source_text}
    {target_lang}:
    """
    
    return prompt


def reflection_prompt_generate(source_lang: str, target_lang: str, source_text: str, translation: str):
    prompt = f"""
    你是一位语言学专家，专门从事从{source_lang}到{target_lang}的翻译工作。
    我会提供一个原文及其翻译，你的目标是改进这个译文。
    你的任务是认真阅读原文及其从{source_lang}到{target_lang}的译文，然后给出改进译文的建设性和有用的建议。
    原文及其译文使用 XML 的标签 <SOURCE_TEXT></SOURCE_TEXT> 和 <TRANSLATION></TRANSLATION> 进行分割，如下：
                
    <SOURCE_TEXT>
    {source_text}
    </SOURCE_TEXT>
                
    <TRANSLATION>
    {translation}
    </TRANSLATION>
                
    在给出建议时，要着眼于是否有方法来改进译文。
    1.准确性（通过纠正误译、遗漏或未翻译的文本）
    2.流畅（根据{target_lang}的语法、拼写以及标点符号的规则进行修改，并消除不必要的重复）
    3.风格（确保译文反映原文的语言风格并考虑源语言的文化背景）
    4.术语（确保术语使用的一致性并反映原文所属的领域，确保使用和{target_lang}一致的习惯用法）
                
    请提供一个具体、有用且建设性的建议清单以改善翻译质量。
    每条建议针对翻译的一个具体部分。
    只输出建议，不要其他。
    """
                
    return prompt


def final_prompt_generate(source_lang: str, target_lang: str, source_text: str, translation: str, reflection: str) -> str:
    prompt = f"""
    你是一位语言学专家，专门从事从{source_lang}到{target_lang}的翻译工作。
    你的任务是认真阅读并改进从{source_lang}到{target_lang}的翻译，在进行修正时需要考虑专家给出的建议。
    原文、初始的译文以及语言专家给出的建议通过 XML 标签 <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> 进行分割，如下：
    <SOURCE_TEXT>
    {source_text}
    </SOURCE_TEXT>

    <TRANSLATION>
    {translation}
    </TRANSLATION>

    <EXPERT_SUGGESTIONS>
    {reflection}
    </EXPERT_SUGGESTIONS>
                
    在优化译文时，需要考虑语言专家给出的建议，并且需要确保如下几点：
    1.准确性（通过纠正误译、遗漏或未翻译的文本）
    2.流畅（根据{target_lang}的语法、拼写以及标点符号的规则进行修改，并消除不必要的重复）
    3.风格（确保译文反映原文的语言风格）
    4.术语（使用上的不一致，和原文的不契合）
    5.其他错误
                
    只输出新的译文，不要其他。
    """
    
    return prompt


def get_response(prompt):
    messages = [ChatMessage(
        role = "user",
        content = prompt
    )]
    
    handler = ChunkPrintHandler()
    result = spark.generate([messages], callbacks=[handler])
    
    return result.generations[0][0].text


if __name__ == '__main__':
    from sparkai.core.callbacks import StdOutCallbackHandler
    spark = ChatSparkLLM(
        spark_api_url=os.environ["SPARKAI_URL"],
        spark_app_id=os.environ["SPARKAI_APP_ID"],
        spark_api_key=os.environ["SPARKAI_API_KEY"],
        spark_api_secret=os.environ["SPARKAI_API_SECRET"],
        spark_llm_domain=os.environ["SPARKAI_DOMAIN"],
        streaming=False,
    )
    
    src_lang = "英文"
    target_lang = "中文"
    
    text = """It was the best of times, it was the worst of times, it was the age of wisdom, 
            it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, 
            it was the season of Light, it was the season of Darkness, it was the spring of hope, 
            it was the winter of despair, we had everything before us, we had nothing before us, 
            we were all going direct to Heaven, we were all going direct the other way—in short, 
            the period was so far like the present period, that some of its noisiest authorities 
            insisted on its being received, for good or for evil, in the superlative degree of 
            comparison only."""
    
    initial_prompt = initial_prompt_generate(src_lang, target_lang, text)
    initial_translation_text = get_response(initial_prompt)
    print(initial_translation_text)
    print("------------------------------------------------------------------------------------------------")
    reflection_prompt = reflection_prompt_generate(src_lang, target_lang, text, initial_translation_text)
    reflection_text = get_response(reflection_prompt)
    print(reflection_text)
    print("------------------------------------------------------------------------------------------------")
    final_prompt = final_prompt_generate(src_lang, target_lang, text, initial_translation_text, reflection_text)
    final_translation_text = get_response(final_prompt)
    print(final_translation_text)
