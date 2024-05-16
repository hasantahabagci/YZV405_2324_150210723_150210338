# @Author: M.Serdar NAZLI and Hasan Taha BAĞCI, Istanbul Technical University. 
# @Date: 27/04/2024 
# Prepared for the NLP project.

mapping = {
    "ğ":"ı", "…": "...", "ğ": "ü", "ã¶": "ö", "ğ":"ö", "å": "ş", "ã§": "ç", "ã¼": "ü", "äŸ": "ğ", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ã¼": "ü", "ã§": "ç", "ã¶": "ö", "ã‡": "Ç", "ã–": "Ö", "ãœ": "Ü", "ã": "ğ", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ", "ä": "Ğ", "ä±": "ı", "ä°": "İ", "ä±": "ı", "ä": "ğ", "ä": "Ğ", "ä": "ğ", "ä": "ğ",
    "ğ¢":"a", "ğ": "ç", "å": "ş"
}


to_removed_chars = ["»", "«", "§", "", "¦", "®", "|", 
                    "<p>", r"<\p>", "<br>", r"<br />", r"<br/>", "•", "¤", "<q>" ,r"<\q>", r"\x92" , r"</q>"
                    "¹", #"#",
                    "", "œ", "∙", "\xad", "\x92", "\r", "\n", 
                    ]



conversion_dict = {' ': 0, 
 '!': 1,
 '"': 2,
 '#': 3,
 '$': 4,
 '%': 5,
 '&': 6,
 "'": 7,
 '(': 8,
 ')': 9,
 '*': 10,
 ',': 11,
 '-': 12,
 '.': 13,
 '/': 14,
 '0': 15,
 '1': 16,
 '2': 17,
 '3': 18,
 '4': 19,
 '5': 20,
 '6': 21,
 '7': 22,
 '8': 23,
 '9': 24,
 ':': 25,
 ';': 26,
 '=': 27,
 '?': 28,
 '@': 29,
 '[': 30,
 ']': 31,
 '_': 32,
 '`': 33,
 'a': 34,
 'b': 35,
 'c': 36,
 'd': 37,
 'e': 38,
 'f': 39,
 'g': 40,
 'h': 41,
 'i': 42,
 'j': 43,
 'k': 44,
 'l': 45,
 'm': 46,
 'n': 47,
 'o': 48,
 'p': 49,
 'q': 50,
 'r': 51,
 's': 52,
 't': 53,
 'u': 54,
 'v': 55,
 'w': 56,
 'x': 57,
 'y': 58,
 'z': 59,
 '{': 60,
 '}': 61,
 '~': 62,
 '°': 63,
 '´': 33,
 '½': 64,
 'ß': 35,
 'á': 34,
 'â': 34,
 'å': 34,
 'ç': 65,
 'è': 38,
 'é': 38,
 'ê': 38,
 'ì': 42,
 'î': 42,
 'ï': 42,
 'ñ': 47,
 'ó': 48,
 'ô': 48,
 'ö': 66,
 'ø': 48,
 'ú': 54,
 'û': 54,
 'ü': 67,
 'ē': 38,
 'ğ': 68,
 'İ': 42,
 'ı': 69,
 'ş': 70,
 'š': 52,
 'ţ': 53,
 '–': 12,
 '—': 12,
 '‘': 7,
 '’': 7,
 '“': 2,
 '”': 2,
 '′': 33,
 '″': 2, 
 "UNK": 71, 
 "[MASK]": 72, 
 "[PAD]": 73, 
 "[EOS]": 75,
 "[SOS]": 74}


reverse_conversion_dict = {}
for key, value in conversion_dict.items():
    reverse_conversion_dict[value] = key

reverse_conversion_dict[2] = '"'
reverse_conversion_dict[12] = "-"
reverse_conversion_dict[7] = "'"
reverse_conversion_dict[33] = "´"
reverse_conversion_dict[34] = "a"
reverse_conversion_dict[35] = "b"
reverse_conversion_dict[38] = "e"
reverse_conversion_dict[42] = "i"
reverse_conversion_dict[43] = "n"
reverse_conversion_dict[47] = "n"
reverse_conversion_dict[48] = "o"
reverse_conversion_dict[52] = "s"
reverse_conversion_dict[53] = "t"
reverse_conversion_dict[54] = "u"



if __name__ == "__main__":
    print("mapping:", mapping)
    print("to_removed_chars:", to_removed_chars)
    print("conversion_dict:", conversion_dict) 
    print("reverse_conversion_dict:", reverse_conversion_dict)