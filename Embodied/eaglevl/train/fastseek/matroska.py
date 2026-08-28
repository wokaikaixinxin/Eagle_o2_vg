from fractions import Fraction

from bitstring.bits import BitsType
from ebmlite import MasterElement, loadSchema
from sortedcontainers import SortedList


def parse_matroska(file: BitsType) -> SortedList:
    schema = loadSchema("matroska.xml")
    doc = schema.load(file, headers=True)

    # Get cue times
    stack = [doc]
    cue_times = SortedList()
    all_names = []
    timescale = 1e6  # Default matroska timescale in ns is 1ms
    duration = None
    while len(stack) > 0:
        el = stack.pop()
        if el.name == "Duration":
            duration = el.value
        all_names.append(el.name)
        if el.name == "CueTime":
            cue_times.add(el.value)
        elif el.name == "TimecodeScale":
            timescale = el.value
        elif isinstance(el, MasterElement):
            stack.extend([c for c in el])
    ns_per_s: int = int(1e9)
    return cue_times, Fraction(timescale, ns_per_s), duration
