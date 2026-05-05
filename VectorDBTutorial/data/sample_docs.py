# ============================================================
#  Sample documents for the RAG / Vector DB demo
#  4 clearly different topics so we can see clusters in PCA
# ============================================================

DOCUMENTS = [

    # ── SPACE (5 docs) ──────────────────────────────────────
    {
        "id": "space_001",
        "text": (
            "The International Space Station orbits Earth at an altitude of roughly 408 kilometres. "
            "It travels at 28,000 km/h, completing 16 full orbits every single day. "
            "Astronauts aboard experience a sunrise every 90 minutes."
        ),
        "metadata": {
            "title": "International Space Station",
            "category": "Space",
            "source": "NASA",
            "year": 2023,
        },
    },
    {
        "id": "space_002",
        "text": (
            "Black holes are regions of spacetime where gravity is so strong that nothing — "
            "not even light — can escape. They form when massive stars collapse at the end of "
            "their lives. The boundary beyond which escape is impossible is called the event horizon."
        ),
        "metadata": {
            "title": "Black Holes",
            "category": "Space",
            "source": "ESA",
            "year": 2022,
        },
    },
    {
        "id": "space_003",
        "text": (
            "NASA's Perseverance rover landed on Mars in February 2021. It is searching for signs "
            "of ancient microbial life, collecting rock samples, and testing oxygen production from "
            "the Martian atmosphere — a key step toward future human missions."
        ),
        "metadata": {
            "title": "Mars Exploration",
            "category": "Space",
            "source": "NASA",
            "year": 2021,
        },
    },
    {
        "id": "space_004",
        "text": (
            "The Big Bang theory states that the universe began as an extremely hot, dense point "
            "approximately 13.8 billion years ago and has been expanding ever since. "
            "Cosmic Microwave Background radiation is the 'echo' of that early hot state."
        ),
        "metadata": {
            "title": "The Big Bang",
            "category": "Space",
            "source": "Hubble",
            "year": 2020,
        },
    },
    {
        "id": "space_005",
        "text": (
            "Our solar system contains eight planets orbiting the Sun. "
            "Jupiter is the largest, with a mass greater than all other planets combined. "
            "Saturn is famous for its stunning ring system made of ice and rock particles."
        ),
        "metadata": {
            "title": "Solar System Planets",
            "category": "Space",
            "source": "NASA",
            "year": 2023,
        },
    },

    # ── ANIMALS (5 docs) ────────────────────────────────────
    {
        "id": "animal_001",
        "text": (
            "The blue whale is the largest animal ever known to have existed on Earth. "
            "It can reach lengths of up to 30 metres and weigh as much as 200 tonnes. "
            "Despite their enormous size, blue whales feed almost exclusively on tiny shrimp-like creatures called krill."
        ),
        "metadata": {
            "title": "Blue Whale",
            "category": "Animals",
            "source": "WWF",
            "year": 2022,
        },
    },
    {
        "id": "animal_002",
        "text": (
            "Honeybees are essential pollinators responsible for one third of the food humans eat. "
            "A single hive can contain up to 60,000 bees. Worker bees communicate the location of "
            "flowers to each other through a precise 'waggle dance'."
        ),
        "metadata": {
            "title": "Honeybees",
            "category": "Animals",
            "source": "National Geographic",
            "year": 2021,
        },
    },
    {
        "id": "animal_003",
        "text": (
            "Dolphins are highly intelligent marine mammals that use echolocation to navigate and hunt. "
            "They produce clicks and listen for the echoes reflecting off objects. "
            "Dolphins live in social groups called pods and are known to display empathy and play behaviour."
        ),
        "metadata": {
            "title": "Dolphins",
            "category": "Animals",
            "source": "Marine Biology Journal",
            "year": 2023,
        },
    },
    {
        "id": "animal_004",
        "text": (
            "Emperor penguins are the tallest and heaviest of all penguin species. "
            "They breed during the Antarctic winter, with male penguins balancing eggs on their feet "
            "for two months in temperatures as low as -60°C while females hunt at sea."
        ),
        "metadata": {
            "title": "Emperor Penguins",
            "category": "Animals",
            "source": "BBC Wildlife",
            "year": 2022,
        },
    },
    {
        "id": "animal_005",
        "text": (
            "The cheetah is the fastest land animal, capable of reaching speeds of 120 km/h in short bursts. "
            "It accelerates from 0 to 100 km/h in just three seconds. "
            "Unlike other big cats, cheetahs cannot roar — they purr and chirp instead."
        ),
        "metadata": {
            "title": "Cheetah",
            "category": "Animals",
            "source": "African Wildlife Foundation",
            "year": 2023,
        },
    },

    # ── COOKING (5 docs) ────────────────────────────────────
    {
        "id": "cooking_001",
        "text": (
            "The Maillard reaction is a chemical reaction between amino acids and sugars that gives "
            "browned food its distinctive flavour. It occurs above 140°C and is responsible for the "
            "crust on bread, the sear on steak, and the golden colour of roasted coffee beans."
        ),
        "metadata": {
            "title": "Maillard Reaction",
            "category": "Cooking",
            "source": "Food Science Weekly",
            "year": 2021,
        },
    },
    {
        "id": "cooking_002",
        "text": (
            "Traditional Italian pasta is made from durum wheat semolina and water. "
            "Different shapes are designed for different sauces — wide flat pappardelle for rich ragù, "
            "hollow rigatoni to trap chunky sauces, and thin spaghetti for light oil-based preparations."
        ),
        "metadata": {
            "title": "Pasta Making",
            "category": "Cooking",
            "source": "Culinary Institute",
            "year": 2022,
        },
    },
    {
        "id": "cooking_003",
        "text": (
            "Fermentation is a metabolic process where microorganisms like bacteria and yeast convert "
            "sugars into acids, gases, or alcohol. It is used to make bread, beer, wine, cheese, yoghurt, "
            "kimchi, and sauerkraut. Fermentation both preserves food and develops complex flavours."
        ),
        "metadata": {
            "title": "Fermentation in Cooking",
            "category": "Cooking",
            "source": "Food Science Weekly",
            "year": 2023,
        },
    },
    {
        "id": "cooking_004",
        "text": (
            "Baking is a precise science. Gluten — formed when flour meets water — provides structure. "
            "Leavening agents like baking soda release CO2 bubbles, making dough rise. "
            "Fat coats gluten strands, keeping cakes tender. Temperature and timing are critical."
        ),
        "metadata": {
            "title": "Science of Baking",
            "category": "Cooking",
            "source": "Culinary Institute",
            "year": 2022,
        },
    },
    {
        "id": "cooking_005",
        "text": (
            "Knife skills are fundamental to cooking efficiency and safety. "
            "The julienne cut produces thin matchstick strips ideal for stir-fries. "
            "Chiffonade is used for leafy herbs — stack the leaves, roll them, and slice into ribbons. "
            "A sharp knife is always safer than a blunt one."
        ),
        "metadata": {
            "title": "Knife Techniques",
            "category": "Cooking",
            "source": "Chef's Handbook",
            "year": 2021,
        },
    },

    # ── TECHNOLOGY (5 docs) ─────────────────────────────────
    {
        "id": "tech_001",
        "text": (
            "Machine learning is a branch of AI where systems learn patterns from data rather than "
            "being explicitly programmed. Neural networks — inspired by the human brain — consist of "
            "layers of nodes that transform input data into predictions through weighted connections."
        ),
        "metadata": {
            "title": "Machine Learning",
            "category": "Technology",
            "source": "MIT OpenCourseWare",
            "year": 2023,
        },
    },
    {
        "id": "tech_002",
        "text": (
            "Blockchain is a distributed ledger technology where data is stored in blocks chained "
            "together cryptographically. No single entity controls the chain. "
            "It underpins cryptocurrencies like Bitcoin and enables trustless smart contracts on platforms like Ethereum."
        ),
        "metadata": {
            "title": "Blockchain Technology",
            "category": "Technology",
            "source": "IEEE",
            "year": 2022,
        },
    },
    {
        "id": "tech_003",
        "text": (
            "Quantum computers use qubits which, unlike classical bits, can exist in a superposition "
            "of 0 and 1 simultaneously. This allows quantum computers to explore many solutions at once. "
            "They are expected to revolutionise cryptography, drug discovery, and optimisation problems."
        ),
        "metadata": {
            "title": "Quantum Computing",
            "category": "Technology",
            "source": "IBM Research",
            "year": 2023,
        },
    },
    {
        "id": "tech_004",
        "text": (
            "The Internet runs on TCP/IP — a suite of protocols that govern how data is split into packets, "
            "routed across networks, and reassembled at the destination. IP handles addressing; "
            "TCP ensures packets arrive in order and without errors."
        ),
        "metadata": {
            "title": "Internet Protocols",
            "category": "Technology",
            "source": "IETF",
            "year": 2021,
        },
    },
    {
        "id": "tech_005",
        "text": (
            "GPUs (Graphics Processing Units) were originally designed for rendering images but are now "
            "essential for AI. Their thousands of small cores excel at parallel matrix operations — "
            "exactly what training neural networks requires. NVIDIA's CUDA platform made GPUs programmable for general computing."
        ),
        "metadata": {
            "title": "GPU Architecture",
            "category": "Technology",
            "source": "NVIDIA Developer Blog",
            "year": 2023,
        },
    },
]

CATEGORIES = ["Space", "Animals", "Cooking", "Technology"]
