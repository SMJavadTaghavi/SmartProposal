
import re
import string


def remove_extra_punctuation(text: str) -> str:
    """
    Emoji, symbol va fasele haye ezafi ro az matn hazf mikone
    """

    # 1. Hazf emoji ha
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoji haye chehreh
        "\U0001F300-\U0001F5FF"  # symbol ha
        "\U0001F680-\U0001F6FF"  # transport va map
        "\U0001F1E0-\U0001F1FF"  # flag ha
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # 2. Hazf symbol va neshanehaye negarshi
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # 3. Hazf fasele haye posht-sar-ham
    text = re.sub(r"\s+", " ", text).strip()

    return text


def count_non_alphabetic(text: str) -> int:
    """
    Shomarande character haye gheyr-harfi
    (harf nabashan va space ham nabashan)
    """
    return sum(1 for c in text if not c.isalpha() and not c.isspace())


# -------------------------------
# Test sade va mostaghel
# -------------------------------

def test_remove_extra_punctuation():
    original_text = "Hello!!!  This 😊😊 is   a test!!! ### $$$"
    cleaned_text = remove_extra_punctuation(original_text)

    before_count = count_non_alphabetic(original_text)
    after_count = count_non_alphabetic(cleaned_text)

    print("Matn asli :", original_text)
    print("Matn tamiz:", cleaned_text)
    print("Gheyr-harfi ghabl:", before_count)
    print("Gheyr-harfi baad  :", after_count)

    # Shart test bar asas tozih issue
    if after_count < before_count:
        print("Test ba movafaghiat anjam shod ")
    else:
        print("Test namovafagh ")


if __name__ == "__main__":
    test_remove_extra_punctuation()
