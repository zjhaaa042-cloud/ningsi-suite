"""量表定义与题目（Zung SAS / SDS 中文版，20 题 4 级计分）。

反向题在计分时按 5 - 作答值反转，粗分乘 1.25 取整得到标准分。
"""

from __future__ import annotations

from dataclasses import dataclass

OPTIONS = ("没有或很少时间", "小部分时间", "相当多时间", "绝大部分或全部时间")

SAS_ITEMS = (
    ("我觉得比平常容易紧张和着急", False),
    ("我无缘无故地感到害怕", False),
    ("我容易心里烦乱或觉得惊恐", False),
    ("我觉得我可能将要发疯", False),
    ("我觉得一切都很好，也不会发生什么不幸", True),
    ("我手脚发抖打颤", False),
    ("我因为头痛、颈痛和背痛而苦恼", False),
    ("我感觉容易衰弱和疲乏", False),
    ("我觉得心平气和，并且容易安静坐着", True),
    ("我觉得心跳得很快", False),
    ("我因为一阵阵头晕而苦恼", False),
    ("我有过晕倒发作，或觉得要晕倒似的", False),
    ("我呼气吸气都感到很容易", True),
    ("我手脚麻木和刺痛", False),
    ("我因为胃痛和消化不良而苦恼", False),
    ("我常常要小便", False),
    ("我的手常常是干燥温暖的", True),
    ("我脸红发热", False),
    ("我容易入睡并且一夜睡得很好", True),
    ("我做噩梦", False),
)

SDS_ITEMS = (
    ("我觉得闷闷不乐，情绪低沉", False),
    ("我觉得一天之中早晨最好", True),
    ("我一阵阵哭出来或觉得想哭", False),
    ("我晚上睡眠不好", False),
    ("我吃得跟平常一样多", True),
    ("我与异性密切接触时和以往一样感到愉快", True),
    ("我发觉我的体重在下降", False),
    ("我有便秘的苦恼", False),
    ("我心跳比平常快", False),
    ("我无缘无故地感到疲乏", False),
    ("我的头脑跟平常一样清楚", True),
    ("我觉得经常做的事情并没有困难", True),
    ("我觉得不安而平静不下来", False),
    ("我对将来抱有希望", True),
    ("我比平常容易生气激动", False),
    ("我觉得作出决定是容易的", True),
    ("我觉得自己是个有用的人，有人需要我", True),
    ("我的生活过得很有意思", True),
    ("我认为如果我死了别人会生活得好些", False),
    ("平常感兴趣的事我仍然照样感兴趣", True),
)


@dataclass(frozen=True)
class Scale:
    code: str
    name: str
    items: tuple
    factor: float
    version: str = "zung-cn-v1"

    @property
    def size(self) -> int:
        return len(self.items)

    def reverse_indices(self) -> tuple:
        return tuple(index + 1 for index, (_, reverse) in enumerate(self.items) if reverse)


SAS = Scale(code="SAS", name="焦虑自评量表", items=SAS_ITEMS, factor=1.25)
SDS = Scale(code="SDS", name="抑郁自评量表", items=SDS_ITEMS, factor=1.25)
SCALES = {SAS.code: SAS, SDS.code: SDS}


def get_scale(code: str) -> Scale:
    try:
        return SCALES[code.upper()]
    except KeyError as error:
        raise KeyError(f"未知量表：{code}") from error
