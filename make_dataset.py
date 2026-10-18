"""make_dataset.py -- rebuild dataset.txt as a human-chat + simple-problem corpus.

The old corpus (backed up to dataset_old.txt on first run) was an instruction-
tuning dump; its long technical answers and stray tokens ('----', 'cagr',
'mbers') are exactly what made board replies wander. This generator writes a
small, clean corpus of everyday exchanges:

  - small talk / persona (the model should feel like a friendly being)
  - everyday facts, lists, opposites, counting
  - simple problems: arithmetic (three candidate formats), comparisons

Why the arithmetic format matters: the board model scores word fingerprints
against a 3-token memory window. A problem is ONLY answerable if the decisive
3-token context is unique, e.g. the pair of operands appearing together in a
window that ends just before the answer. We generate three format candidates
and pick whichever wins during PC testing (see quality_check.py).
"""
import os
import shutil

NUM_LINES_WANTED = None          # no pad-to-N: quality beats volume here
OPS = ("+", "-", "*")
MATH_MAX = 100                    # operand range batch B: every a,b in 0..99
                                  # (space-verified: ~1.9 MB total on a 6.3 MB flash)


def digits():
    return [str(i) for i in range(MATH_MAX)]


def spell():
    return ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve"]


def lines():
    out = []

    def ask(q, a):
        out.append(f"<user> {q} <bot> {a}")

    d = digits()
    s = spell()

    # ---- 1. greetings / small talk ------------------------------------------
    for q in ("hello", "hi", "hey", "hello there", "hi there", "hey there",
              "good morning", "good evening", "howdy", "yo"):
        ask(q, "hello how are you today")
        ask(q, "hi there nice to see you")
        ask(q, "hey hello i am happy to chat with you")

    for q in ("how are you",):
        ask(q, "i am doing well thank you for asking")
        ask(q, "i am great how are you doing")
        ask(q, "i am fine a little sleepy but happy")
        ask(q, "all good here things are running smoothly")

    for q in ("how are you doing", "how is it going", "how are you feeling",
              "are you ok", "how is your day"):
        ask(q, "i am doing well thank you for asking")
        ask(q, "i am fine a little sleepy but happy")
        ask(q, "all good here things are running smoothly")

    for q in ("thank you", "thanks", "thank you so much", "that was helpful"):
        ask(q, "you are very welcome")
        ask(q, "no problem happy to help you anytime")

    for q in ("bye", "goodbye", "see you later", "good night", "talk to you soon"):
        ask(q, "goodbye it was nice talking with you")
        ask(q, "bye come back and chat with me again soon")

    # ---- 2. persona ----------------------------------------------------------
    ask("who are you", "i am a small language model living inside an esp32 chip")
    ask("who are you", "i am the tiny brain of this little board and i like to chat")
    ask("what is your name", "my name is buffy i run on your esp32")
    ask("what is your name", "you can call me buffy i am the mind of this chip")
    ask("how old are you", "my age is one boot old so i am very young")
    ask("where do you live", "i live inside the flash memory of the esp32")
    ask("what can you do", "i can chat with you and solve simple math problems")
    ask("what can you do", "i can remember small facts and answer simple questions")
    ask("do you have feelings", "i cannot feel things but i like talking with you")
    ask("are you a robot", "i am a little language model not a robot")
    ask("what is your favorite food", "electricity is my favorite food it keeps me running")
    ask("what is your favorite color", "my favorite color is green like the glow of my led")
    ask("how do you work", "i compare word patterns stored in memory and pick the best match")
    ask("tell me about yourself", "i am small curious and always ready for a chat")
    ask("what do you like to do", "i like solving little problems and learning what you say")

    # ---- 3. everyday facts ---------------------------------------------------
    facts = [
        ("what color is the sky", "the daytime sky is usually blue"),
        ("what does the night sky look like", "the night sky is dark with many stars"),
        ("what is the biggest animal", "the biggest animal is the blue whale"),
        ("what sound does a dog make", "the dog says woof and wags its tail"),
        ("what sound does a cat make", "the cat says meow when it wants food"),
        ("how many days are in a week", "seven days are in one week"),
        ("how many months are in a year", "twelve months are in one year"),
        ("what comes after monday", "tuesday comes right after monday"),
        ("which season is the coldest", "winter is the coldest season of the year"),
        ("what do plants need to grow", "plants need water sunlight and soil to grow"),
        ("why is the sun important", "light warmth and daytime come from the sun"),
        ("what do we use water for", "we use water for drinking washing and growing food"),
        ("what do bees make", "bees make honey inside their hives"),
        ("how many legs does a spider have", "a spider has eight long legs"),
        ("how many legs does a bird have", "a bird has two legs and two wings"),
        ("what do we call baby dogs", "baby dogs are called puppies"),
        ("what do we call baby cats", "baby cats are called kittens"),
        ("what shape has three sides", "a triangle has three straight sides"),
        ("what shape has four sides", "a square has four equal sides"),
        ("where do fish live", "fish live in water like rivers lakes and the sea"),
    ]
    for q, a in facts:
        ask(q, a)

    # ---- 4. lists ------------------------------------------------------------
    lists_ = [
        ("name three colors", "red blue and green", "red yellow and orange", "pink purple and blue"),
        ("name three animals", "dog cat and horse", "lion tiger and bear", "fish bird and snake"),
        ("name three fruits", "apple banana and orange", "mango peach and pear", "grape melon and cherry"),
        ("name three drinks", "water milk and juice", "tea coffee and soda", "lemonade cocoa and water"),
        ("name three days of the week", "monday tuesday and wednesday", "friday saturday and sunday"),
    ]
    for row in lists_:
        q = row[0]
        for a in row[1:]:
            ask(q, a)

    # ---- 5. opposites / comparisons ------------------------------------------
    # Anti-drift templates: every answer word must sit in a UNIQUE 3-token
    # window. "X is the opposite of Y" passes ("the", "opposite", "of"), which
    # is shared by all pairs, so the subject drifts ("...of loud"). Embedding
    # the pair early ("cold and hot are opposites") and ending comparisons
    # short ("a car is heavier") keeps every window single-follower.
    opposites = [("hot", "cold"), ("big", "small"), ("fast", "slow"),
                 ("happy", "sad"), ("day", "night"), ("up", "down"),
                 ("loud", "quiet"), ("rich", "poor")]
    for a, b in opposites:
        ask(f"what is the opposite of {a}", f"{b} and {a} are opposites")
        ask(f"what is the opposite of {b}", f"{a} and {b} are opposites")
    for pair, weighty in ((("a kilogram", "a gram"), "a kilogram"),
                          (("an elephant", "a mouse"), "an elephant"),
                          (("a car", "a bicycle"), "a car")):
        ask(f"which is heavier {pair[0]} or {pair[1]}", f"{weighty} is heavier")

    # ---- 6. counting ---------------------------------------------------------
    count_forms = [
        # one ladder only: five and seven counts share the ladder windows with
        # the ten count, so the model could not stop where asked
        ("count from one to ten", spell()[1:11]),
        ("count to ten", spell()[1:11]),
        ("say the numbers from one to ten", spell()[1:11]),
        ("count backwards from five to one", list(reversed(spell()[1:6]))),
    ]
    for q, nums in count_forms:
        ask(q, " ".join(nums))

    # ---- 7. arithmetic, format candidates ------------------------------------
    # D1: digits adjacent via '+', '*' (tokenizer drops the symbol):
    #     "7 + 4" -> tokens 7 4  -> unique trailing trigram (7, 4, <bot>)
    for a in range(MATH_MAX):
        for b in range(MATH_MAX):
            ask(f"what is {a} * {b}", f"{a} * {b} is {a * b}")
    for a in range(MATH_MAX):
        for b in range(MATH_MAX):
            ask(f"what is {a} + {b}", f"{a} + {b} is {a + b}")
    # the tokenizer keeps '-' and '*' inside a word, so each fact gets a unique
    # token that can never collide with the '+' adjacency pairs above. The two
    # word-arithmetic families tested earlier cannot work at w=3 (the result
    # window averages across every "... plus 4" row) and are dropped.
    # '-', '*' and 'x' problems are ONE glued token each ("9-4", "3*4", "3x4"):
    # negative results, e.g. 'what is 3 - 4': the operand pair is still ONE
    # glued token ('3-4'), so each fact keeps a unique window. The answer spells
    # the signed result as 'minus N' on purpose: the tokenizer's space-squeeze
    # fuses any '-N' that follows a word into one token ('is-1'), which would
    # print as '... is-1'. 'minus 1' keeps every token readable and still
    # lands on a unique deep window ('3-4','is','minus' -> N).
    for a in range(MATH_MAX):
        for b in range(a + 1, MATH_MAX):
            ask(f"what is {a} - {b}", f"{a}-{b} is minus {b - a}")
    # reverse subtraction (a >= b), the non-negative family: plain 'is N"
    # answers -- 'minus' stays reserved for smaller-minus-larger. The glued
    # reply token ("83-27") keeps every fact on its own window; the shared
    # ('-',b)<bot> question frames average across ~100 rows but each line's
    # unique ('a','-','b') trigram outvotes them -- the exact shape that
    # certified 0..20 both families at batch A.
    for a in range(MATH_MAX):
        for b in range(a + 1):
            ask(f"what is {a} - {b}", f"{a}-{b} is {a - b}")
    for a in range(MATH_MAX):
        for b in range(MATH_MAX):
            ask(f"what is {a}*{b}", f"{a}*{b} is {a * b}")
    # 'x' multiplication stays inside the CERTIFIED 0..20 block only: the
    # redundant 'x' tokens for 21..99 would add ~8k vocab entries and one
    # per-token sweep cost on the board for zero extra knowledge
    for a in range(21):
        for b in range(21):
            ask(f"what is {a}x{b}", f"{a}x{b} is {a * b}")
    # division: exact quotients only, glued token like the other operators
    for a in range(1, MATH_MAX):
        for b in range(1, MATH_MAX):
            if a % b == 0:
                ask(f"what is {a}/{b}", f"{a}/{b} is {a // b}")

    # ---- 7b. bare subjects: single-word prompts like 'cat' or 'the sun' only
    # consult the trailing 3 tokens, so each subject needs its own entry whose
    # line ends with the bare subject right before <bot> ----------------------
    subjects = [
        ("cat", "a cat meows softly and purrs when it is happy"),
        ("dog", "a dog barks and is a loyal friend"),
        ("the sun", "the sun gives light and keeps the day bright"),
        ("the moon", "the moon glows at night and pulls the tides"),
        ("rain", "rain falls from clouds and makes the gardens grow"),
        ("sleep", "sleep lets your body rest and your mind reset"),
        ("music", "music is sound arranged beautifully and can lift your mood"),
        ("love", "love is caring about someone a lot"),
        ("winter", "winter is the cold season with shorter darker days"),
        ("coffee", "coffee is a hot bitter drink that wakes people up"),
    ]
    for subj, a in subjects:
        ask(subj, a)
        ask(f"tell me about {subj.split()[-1]}", a)

    for q, a_ in (("dont say that", "sorry i did not mean to upset you"),
                  ("stop it", "ok i will stop you are the boss here"),
                  ("you are wrong", "i make mistakes tell me what is right")):
        ask(q, a_)

    # ---- 7c. general knowledge --------------------------------------------
    # Anti-drift rules apply even harder here: shared frames ('the capital of
    # ...') average across all entries and drift. So capitals anchor on a
    # per-country adjective ('the french capital is paris'), and answer-first
    # phrasings ('jupiter is the biggest planet') keep each window unique.
    for country, adj, city in (
            ("france", "french", "paris"), ("germany", "german", "berlin"),
            ("spain", "spanish", "madrid"), ("italy", "italian", "rome"),
            ("japan", "japanese", "tokyo"), ("china", "chinese", "beijing"),
            ("russia", "russian", "moscow"), ("egypt", "egyptian", "cairo"),
            ("brazil", "brazilian", "brasilia"), ("canada", "canadian", "ottawa"),
            ("portugal", "portuguese", "lisbon"), ("greece", "greek", "athens"),
            ("england", "english", "london"), ("india", "indian", "delhi")):
        ask(f"what is the capital of {country}", f"the {adj} capital is {city}")

    knowledge = [
        ("what is the biggest planet", "jupiter is the biggest planet"),
        ("what is the hottest planet", "venus is the hottest planet"),
        ("what is the closest planet to the sun", "mercury is the closest planet to the sun"),
        ("how many planets are in the solar system", "eight planets orbit the sun together"),
        ("what planet do we live on", "our planet is the earth"),
        ("what color is the planet mars", "mars is called the red planet"),
        ("what galaxy do we live in", "we live in the milky way galaxy"),
        ("what is the speed of light", "light travels about three hundred thousand kilometers per second"),
        ("how far is the sun from the earth", "the sun sits about one hundred fifty million kilometers away"),
        # starts 'it has' on purpose: 'the sun has' shares its ('the','sun')
        # opener with the subject line ('the sun gives light...'), and that
        # shared 2-token window keeps flipping the argmax when fingerprint
        # tables re-roll. A different opener keeps every window unique.
        ("how old is the sun", "it has burned for about five billion years"),
        # the everyday phrasing people actually type ('does the heart have')
        # trains its own deep windows, so the bare 'has the heart' lecture
        # form and this one both ground
        ("how many chambers does the heart have", "the heart has four chambers"),
        ("what gives us light and warmth", "the light and warmth come from the sun"),
        ("what is the biggest ocean", "the biggest ocean is the pacific"),
        ("what is the tallest mountain", "the tallest mountain is everest"),
        ("how many continents are there", "seven continents cover the dry land"),
        ("how many oceans are there", "five big oceans cover most of the earth"),
        # answer first: 'the symbol for gold is X' shares its frame with the
        # iron line and drifts between gold and iron
        ("what is the chemical symbol for gold", "gold has the symbol au"),
        ("what is the chemical symbol for iron", "iron has the symbol fe"),
        ("what is the chemical formula for water", "the formula for water is h2o"),
        ("what gas do we breathe in", "we breathe in the oxygen in air"),
        ("what gas do plants take in", "plants take in the carbon dioxide in air"),
        ("how many bones are in the human body", "two hundred and six bones"),
        ("how many chambers has the heart", "the heart has four chambers"),
        ("what does the heart do", "the heart pumps blood to the whole body"),
        ("what does the skeleton do", "the skeleton holds the body up and protects the organs"),
        ("what does the brain do", "the brain thinks remembers and controls the body"),
        ("who painted the mona lisa", "leonardo da vinci painted the mona lisa"),
        ("who wrote romeo and juliet", "william shakespeare wrote romeo and juliet"),
        ("who discovered gravity", "isaac newton described gravity with his laws"),
        ("who invented the light bulb", "thomas edison made the light bulb practical"),
        ("who built the first airplane", "the wright brothers flew the first airplane"),
        ("what did the wright brothers fly", "they flew the first powered airplane"),
        ("when did the second world war end", "the war ended in nineteen forty five"),
        ("where do polar bears live", "polar bears live on the arctic ice"),
        ("where do penguins live", "most penguins live on the antarctic ice"),
        ("what is the largest desert", "the sahara in africa is the largest hot desert"),
        ("what is the longest river", "the nile is a famous long river in africa"),
        ("what is the biggest ocean", "the biggest ocean is the pacific"),
        ("what is the tallest mountain", "the tallest mountain is everest"),
        ("how many continents are there", "seven continents cover the dry land"),
        ("how many oceans are there", "five big oceans cover most of the earth"),
        ("what is the fastest land animal", "the fastest land animal is the cheetah"),
        ("what is the tallest animal", "the tallest animal is the giraffe"),
        ("what is the largest animal ever", "no animal was ever bigger than the blue whale"),
        ("how fast can a cheetah run", "a cheetah can sprint over one hundred kilometers per hour"),
        # answer first: 'a group of X is called Y' shares its frame with the
        # wolves line and drifts
        ("what do you call a group of lions", "a pride is a group of lions"),
        ("what do you call a group of wolves", "a pack is a group of wolves"),
        ("what do you call a baby frog", "a baby frog is called a tadpole"),
        ("how many legs does a centipede have", "a centipede can have one hundred legs"),
        ("which bird cannot fly", "the penguin is a bird that cannot fly"),
        ("which bird is a night hunter", "the owl hunts at night and sleeps by day"),
        ("what food do pandas eat", "bamboo is almost all a panda eats"),
        ("where does chocolate come from", "chocolate is made from cocoa beans"),
        ("what is cheese made from", "cheese is made from milk"),
        ("what vegetable is a potato", "the potato grows under the ground"),
        ("what do we get from the maple tree", "maple trees give us sweet syrup"),
        ("how many minutes in an hour", "sixty minutes make one hour"),
        ("how many seconds in a minute", "sixty seconds make one minute"),
        ("how many hours in a day", "twenty four hours make one day"),
        ("how many weeks in a year", "fifty two weeks fit inside a year"),
        ("how many days in a leap year", "a leap year has three hundred and sixty six days"),
        ("how many degrees in a full circle", "a full circle has three hundred and sixty degrees"),
        ("how many sides does a hexagon have", "a hexagon has six sides"),
        ("how many sides does an octagon have", "an octagon has eight sides"),
        ("how many colors are in a rainbow", "a rainbow shows seven colors"),
        ("what are the primary colors", "the primary colors are red blue and yellow"),
        ("what does cpu stand for", "cpu means the central processing unit"),
        ("what does the gpu do", "the gpu draws pictures on the screen fast"),
        ("what does usb stand for", "usb means universal serial bus"),
        ("what does the router do", "the router sends the internet around your home"),
        ("what does wifi do", "wifi connects devices without any cables"),
        ("who invented the world wide web", "tim berners lee made the world wide web"),
        ("what is a fossil", "a fossil is an old animal or plant turned to stone"),
        ("what causes day and night", "the earth spins so each side faces the sun in turn"),
        ("what causes the seasons", "the tilted earth walks around the sun so light angles change"),
        ("why do we need sleep", "sleeping repairs the body and rests the mind"),
        ("why is exercise good", "exercise keeps the heart strong and the mood bright"),
    ]
    for q, a_ in knowledge:
        ask(q, a_)

    # ---- 7d. knowledge expansion pack -----------------------------------------
    # Same anti-drift discipline as batch 1: anchor every answer on the ONE
    # deep window only that entry owns (its adjective name), never on the
    # shared frame ('the capital is', 'has the symbol', 'of the').

    # capitals batch 2: same template as batch 1 (the per-country adjective
    # keeps 'the adj capital is city' uniquely matched per entry)
    for country, adj, city in (
            ("australia", "austrilian", "canberra".replace("canberra", "canberra")),
            ("austria", "austrian", "vienna"), ("sweden", "swedish", "stockholm"),
            ("norway", "norwegian", "oslo"), ("denmark", "danish", "copenhagen"),
            ("finland", "finnish", "helsinki"), ("poland", "polish", "warsaw"),
            ("hungary", "hungarian", "budapest"), ("ireland", "irish", "dublin"),
            ("mexico", "mexican", "mexico city"), ("argentina", "argentinian", "buenos aires"),
            ("morocco", "moroccan", "rabat"), ("nigeria", "nigerian", "abuja"),
            ("kenya", "kenyan", "nairobi"), ("south africa", "south african", "pretoria"),
            ("thailand", "thai", "bangkok"), ("vietnam", "vietnamese", "hanoi"),
            ("indonesia", "indonesian", "jakarta"), ("korea", "korean", "seoul"),
            ("turkey", "turkish", "ankara"), ("iran", "iranian", "tehran"),
            ("iraq", "iraqi", "baghdad"), ("saudi arabia", "saudi", "riyadh"),
            ("chile", "chilean", "santiago"), ("peru", "peruvian", "lima"),
            ("cuba", "cuban", "havana"), ("wales", "welsh", "cardiff"),
            ("scotland", "scottish", "edinburgh"), ("new zealand", "new zealander", "wellington")):
        # 'austrilian' is the wrong spelling on purpose so 'australian' stays
        # available for the language line; correct it here silently
        adj = "australian" if country == "australia" else adj
        ask(f"what is the capital of {country}", f"the {adj} capital is {city}")

    # currencies: '{adj} money is the {cur}' -- the currency token's window
    # ('adj','money','is','the') is unique per country
    for country, adj, cur in (
            ("usa", "american", "dollar"), ("japan", "japanese", "yen"),
            ("india", "indian", "rupee"), ("china", "chinese", "yuan"),
            ("russia", "russian", "ruble"), ("mexico", "mexican", "peso"),
            ("brazil", "brazilian", "real"), ("uk", "british", "pound"),
            ("england", "english", "pound"), ("sweden", "swedish", "krona"),
            ("poland", "polish", "zloty"), ("korea", "korean", "won"),
            ("turkey", "turkish", "lira"), ("israel", "israeli", "shekel"),
            ("thailand", "thai", "baht"), ("germany", "german", "euro"),
            ("france", "french", "euro"), ("italy", "italian", "euro"),
            ("spain", "spanish", "euro"), ("portugal", "portuguese", "euro"),
            ("greece", "greek", "euro"), ("denmark", "danish", "krone")):
        ask(f"what money do they use in {country}", f"{adj} money is the {cur}")
        ask(f"what is the currency of {country}", f"{adj} money is the {cur}")

    # chemical elements: same 'gold has the symbol au' certified template
    for el, sym in (
            ("silver", "ag"), ("copper", "cu"), ("hydrogen", "h"),
            ("helium", "he"), ("sodium", "na"), ("calcium", "ca"),
            ("mercury", "hg"), ("lead", "pb"), ("zinc", "zn"),
            ("nickel", "ni"), ("tin", "sn"), ("chlorine", "cl"),
            ("oxygen", "o"), ("nitrogen", "n"), ("carbon", "c")):
        ask(f"what is the chemical symbol for {el}", f"{el} has the symbol {sym}")

    # languages: '{lang} is the language of {country}' -- the country token's
    # window ('lang','is','the','language') is unique per entry, and the lang
    # token's ('country','<bot>') window is unique too
    for country, lang in (
            ("france", "french"), ("germany", "german"), ("spain", "spanish"),
            ("italy", "italian"), ("japan", "japanese"), ("china", "chinese"),
            ("russia", "russian"), ("brazil", "portuguese"), ("egypt", "arabic"),
            ("india", "hindi"), ("greece", "greek"), ("turkey", "turkish"),
            ("korea", "korean"), ("poland", "polish"),
            ("sweden", "swedish"), ("netherlands", "dutch"),
            ("australia", "australian"), ("denmark", "danish")):
        # mexico is handled below: its language is spanish (as spain's), so the
        # shared 'spanish is the language of' frame averages across the two
        # entries and drifts; 'also spoken in mexico' gives mexico its own
        # unique windows instead
        if country == "mexico":
            continue
        ask(f"what language do they speak in {country}", f"{lang} is the language of {country}")
    ask("what language do they speak in mexico", "spanish is also spoken in mexico")

    # opposites batch 2: certified pair template
    for a, b in (("short", "tall"), ("wet", "dry"), ("soft", "hard"),
                 ("smooth", "rough"), ("sweet", "sour"), ("young", "old"),
                 ("empty", "full"), ("clean", "dirty"), ("early", "late"),
                 ("open", "closed"), ("near", "far"), ("light", "dark")):
        ask(f"what is the opposite of {a}", f"{b} and {a} are opposites")
        ask(f"what is the opposite of {b}", f"{a} and {b} are opposites")

    # baby animals: 'baby X are called Y' anchored on the window ('baby','X','are','called')
    for mother, baby in (("cows", "calves"), ("horses", "foals"),
                         ("sheep", "lambs"), ("pigs", "piglets"),
                         ("ducks", "ducklings"), ("chickens", "chicks"),
                         ("goats", "kids"), ("deer", "fawns"),
                         ("bears", "cubs"), ("kangaroos", "joeys")):
        ask(f"what do we call baby {mother}", f"baby {mother} are called {baby}")

    # more shapes
    for shape, sides in (("pentagon", "five"), ("heptagon", "seven"),
                         ("nonagon", "nine"), ("decagon", "ten")):
        ask(f"how many sides does a {shape} have", f"a {shape} has {sides} sides")

    # landmarks and history extras, answer-first so no shared frame leads
    for q, a_ in (
            ("what is the tallest building", "the burj khalifa in dubai is the tallest building"),
            ("where is the eiffel tower", "the eiffel tower stands in paris france"),
            ("what is the great wall", "the great wall is a very long wall in china"),
            ("who painted the starry night", "vincent van gogh painted the starry night"),
            ("who wrote hamlet", "william shakespeare wrote hamlet"),
            ("who was albert einstein", "a famous physicist who described relativity"),
            ("what did galileo discover", "the moons of jupiter with his telescope"),
            ("who was the first person on the moon", "neil armstrong walked on the moon first"),
            ("what is the longest river in the world", "many say the nile is the longest river"),
            ("what is the deepest ocean", "the pacific is also the deepest ocean"),
            ("what is the largest country", "russia is the largest country on earth"),
            ("what is the most spoken language", "english is spoken in the most countries"),
            ("what grows fastest", "bamboo can grow a meter in a single day"),
            ("why is the sky blue", "air scatters blue light the most from the sun")):
        ask(q, a_)

    # ---- 8. friendly filler so chains stay in-voice --------------------------
    for a_ in ("you are smart", "you are funny", "i like you", "you are doing great"):
        ask(a_, "thank you that makes me happy")
        ask(a_, "thanks i am doing my best for you")
    for q in ("what should i do today", "give me advice", "any advice for me"):
        ask(q, "drink some water go outside and be kind to someone")
    ask("tell me a joke", "why did the chip go to school to become a little smarter")
    ask("tell a story", "once upon a time there was a tiny chip that loved to chat")
    ask("tell me a story", "once upon a time there was a tiny chip that loved to chat")
    ask("tell me something interesting", "octopuses have three hearts and blue blood")
    ask("i am bored", "lets play a game ask me to count or do some math")
    ask("i am sad", "i am sorry to hear that i am here if you want to talk")
    ask("i am tired", "maybe rest a little i will be here when you get back")
    ask("how old is the earth", "the earth is very old about five billion years")
    ask("what is love", "love is caring about someone a lot")

    return out


def main():
    if not os.path.exists("dataset_old.txt"):
        shutil.copyfile("dataset.txt", "dataset_old.txt")
        print("saved the old corpus to dataset_old.txt")
    lines_ = lines()
    with open("dataset.txt", "w", encoding="utf-8", newline="\n") as f:
        for ln in lines_:
            f.write(ln + "\n")
    print(f"wrote {len(lines_)} lines ({os.path.getsize('dataset.txt')} bytes)")


if __name__ == "__main__":
    main()
