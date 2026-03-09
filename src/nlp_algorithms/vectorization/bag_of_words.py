def read_text(filename):
    d = dict()
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            all_words = line.split()
            for word in all_words:
                word = word.lower()
                d[word] = d.get(word, 0) + 1
    return d


if __name__ == "__main__":
    d = read_text("../datasets/tiny_shakespear/tiny_shakespeare.txt")
    print(d)
