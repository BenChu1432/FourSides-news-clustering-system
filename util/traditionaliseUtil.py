from opencc import OpenCC

def simplified_to_traditional(text: str) -> str:
    cc = OpenCC('s2t')  # 's2t' = Simplified to Traditional
    return cc.convert(text)