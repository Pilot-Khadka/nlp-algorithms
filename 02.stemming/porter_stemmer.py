import re


class PorterStemmer:
    """
    References:
    original link:  https://tartarus.org/martin/PorterStemmer/
    algorithm:      https://vijinimallawaarachchi.com/2017/05/09/porter-stemming-algorithm/

    simple rule based stemmer, cruder than lemmatization.
    tends to produce stems that are not actual words
    """

    def __init__(self, word):
        self.word = word.lower()

    def step1a(self):
        if self.word.endswith("sses"):
            self.word = re.sub(r"sses$", "ss", self.word)
        elif self.word.endswith("ies"):
            self.word = re.sub(r"ies$", "i", self.word)
        elif self.word.endswith("s") and not self.word.endswith("ss"):
            self.word = re.sub(r"s$", "", self.word)

    def m(self):
        # find all vowel-consonant pair
        return len(re.findall(r"[aeiou][^aeiou]", self.word))

    def contains_vowel(self):
        for char in self.word:
            if char in "aeiou":
                return True
        return False

    def step1b(self):
        if self.word.endswith("eed"):
            original_word = self.word
            self.word = re.sub(r"eed$", "ee", self.word)

            if not self.m() > 0:
                self.word = original_word

        elif self.word.endswith("ed"):
            original_word = self.word
            self.word = re.sub(r"ed$", "", self.word)
            if not self.contains_vowel():
                self.word = original_word
                # if sucessful perform step1b_2
                self.step1b_2()
        elif self.word.endswith("ing"):
            original_word = self.word
            self.word = re.sub(r"ing$", "", self.word)
            if not self.contains_vowel():
                self.word = original_word
                # if sucessful perform step1b_2
                self.step1b_2()

    def double_consonant(self):
        if len(self.word) < 2:
            return False
        if self.word[-1] == self.word[-2] and self.word[-1] not in "aeiou":
            return True
        return False

    def cvc(self):
        if len(self.word) < 3:
            return False

        if (
            self.word[-1] not in "aeiou"
            and self.word[-2] in "aeiou"
            and self.word[-3] not in "aeiou"
            and self.word[-1] not in "wxy"
        ):
            return True
        return False

    def step1b_2(self):
        if self.word.endswith("at"):
            self.word = re.sub(r"at$", "ate", self.word)
        elif self.word.endswith("bl"):
            self.word = re.sub(r"bl$", "ble", self.word)
        elif self.word.endswith("iz"):
            self.word = re.sub(r"iz$", "ize", self.word)
        elif self.word[-1] not in "lsz" and self.double_consonant():
            # *d -> string ends with double consonant
            self.word = self.word[:-1]
        elif self.m() == 1 and self.cvc():
            # *o -> stem ends in cvc
            self.word += "e"

    def step1c(self):
        if self.contains_vowel() and self.word.endswith("y"):
            self.word = re.sub(r"y$", "i", self.word)

    def step2(self):
        suffix_map = {
            "ational": "ate",
            "tional": "tion",
            "enci": "ence",
            "anci": "ance",
            "izer": "ize",
            "abli": "able",
            "alli": "al",
            "entli": "ent",
            "eli": "e",
            "ousli": "ous",
            "ization": "ize",
            "ation": "ate",
            "ator": "ate",
            "alism": "al",
            "iveness": "ive",
            "fulness": "ful",
            "ousness": "ous",
            "aliti": "al",
            "iviti": "ive",
            "biliti": "ble",
        }

        for suffix in suffix_map:
            if self.word.endswith(suffix):
                original_word = self.word
                self.word = self.word[: -len(suffix)] + suffix_map[suffix]

                if self.m() > 0:
                    break
                else:
                    self.word = original_word

    def step3(self):
        suffix_map = {
            "icate": "ic",
            "ative": "",
            "alize": "al",
            "iciti": "ic",
            "ical": "ic",
            "ful": "",
            "ness": "",
        }
        for suffix in suffix_map:
            if self.word.endswith(suffix):
                original_word = self.word
                self.word = self.word[: -len(suffix)] + suffix_map[suffix]

                if self.m() > 0:
                    break
                else:
                    self.word = original_word

    def step4(self):
        suffix_map = [
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ion",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        ]
        for suffix in suffix_map:
            if self.word.endswith(suffix):
                original_word = self.word
                self.word = self.word[: -len(suffix)]

                if self.m() > 0:
                    break
                else:
                    self.word = original_word

    def step5a(self):
        if self.word.endswith("e"):
            original_word = self.word
            self.word = self.word[:-1]

            if not self.m() > 1:
                self.word = original_word

        elif self.cvc() and self.word.endswith("l"):
            original_word = self.word
            self.word = self.word[:-1]

            if not self.m() == 1:
                self.word = original_word

    def step5b(self):
        original_word = self.word
        if self.double_consonant() and self.word.endswith("l"):
            self.word = self.word[:-1]
        if not self.m() > 1:
            self.word = original_word

    def stem(self):
        self.step1a()
        self.step1b()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5a()
        self.step5b()
        print(self.word + " ", end="")


def porter_stemmer(sentences):
    words = sentences.split()

    res = []
    for word in words:
        word = re.sub("[^A-Za-z0-9]+", " ", word)  # remove special characters
        res.append(PorterStemmer(word).stem())
    return res


def main():
    text = "This was not the map we found in Billy Bones’s chest, but an accurate copy, complete in all things names and heights and soundings with the single exception of the red crosses and the written notes."
    # text = "Multidimensional characterization"
    # text = "found"
    # text = "thing"
    porter_stemmer(text)
    # print(m("tree"))
    # print(m("trouble"))
    # print(m("troubles"))


main()
