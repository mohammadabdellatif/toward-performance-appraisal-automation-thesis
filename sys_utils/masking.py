import random
from typing import Iterable


def segment(alphabet: Iterable, segment_size: int) -> list:
    if alphabet is None or len(alphabet) == 0:
        raise ValueError('alphabet cannot be empty')
    segments = []
    for i in range(0, len(alphabet), segment_size):
        segments.append(alphabet[i:i + segment_size])
    return segments


def shift(value: Iterable, steps: int) -> str:
    steps = steps if steps <= len(value) else steps - len(value)
    return value[-steps:] + value[:-steps]


def digit_at(value: int, at: int) -> int:
    value_as_str = str(value)
    return int(value_as_str[-(at + 1)])


class Masking:
    __alphabet: str = 'abcdefghijklmnopqrstuvwxyz0123456789-_'

    def __init__(self,
                 pin_code: int,
                 random_seed: int,
                 alphabet: str = __alphabet) -> None:
        if pin_code > 999 or pin_code < 100:
            raise ValueError('pin_code must be between 100 and 999')
        self.seg_size = digit_at(pin_code, 2)
        self.seg_shift = digit_at(pin_code, 1)
        self.seg_shift = self.seg_shift if self.seg_shift < self.seg_size else self.seg_shift - self.seg_size
        self.random_seed = random_seed
        self.shift = digit_at(pin_code, 0)
        segments = segment(alphabet, self.seg_size)
        random.Random(self.random_seed).shuffle(segments)
        self.alphabet = ''
        for seg in segments:
            self.alphabet += shift(seg, self.seg_shift)
        if self.alphabet == alphabet:
            raise ValueError('Weak pin code')

    def mask(self, value: str):
        return self.__mask(self.shift, value)

    def unmask(self, value: str) -> str:
        return self.__mask(-self.shift, value)

    def __mask(self, the_shift, value):
        if value is None:
            return None
        result = ''
        for c in value.lower():
            try:
                c_idx = self.alphabet.index(c)
            except ValueError:
                result += c
            else:
                e_idx = (the_shift + c_idx) % len(self.alphabet)
                result += self.alphabet[e_idx]
        return result


if __name__ == '__main__':
    masking = Masking(555, 1234)
    print(f'[{masking.alphabet}]')
    print(f">{masking.mask('Hello this is mohammad')}<")
    print(masking.unmask('rgvvy orsn sn wyr9ww9f'))
