"""Extended ImageNet class labels for image classification."""

LABELS = {
    # New labels
    190: "Norfolk terrier",
    206: "golden retriever",
    273: "dingo",
    355: "llama",
    639: "balance beam",
    # Previous labels
    45: "great grey owl",
    74: "hummingbird",
    284: "tiger beetle",
    702: "traffic light",
    753: "racquet",
    # Animals
    39: "swordfish",
    227: "kite",
    318: "cucumber",
    831: "suit",
    938: "broccoli",
    0: {"label": "tench", "scientific_name": "Tinca tinca", "habitat": "Freshwater lakes and rivers", "translations": {"es": "tenca", "fr": "tanche"}},
    1: {"label": "goldfish", "scientific_name": "Carassius auratus", "habitat": "Ponds and aquariums", "translations": {"es": "pez dorado", "fr": "poisson rouge"}},
    2: {"label": "great white shark", "scientific_name": "Carcharodon carcharias", "habitat": "Oceans", "translations": {"es": "tiburón blanco", "fr": "grand requin blanc"}},
    3: {"label": "tiger shark", "scientific_name": "Galeocerdo cuvier", "habitat": "Oceans", "translations": {"es": "tiburón tigre", "fr": "requin tigre"}},
    4: {"label": "hammerhead shark", "scientific_name": "Sphyrnidae", "habitat": "Oceans", "translations": {"es": "tiburón martillo", "fr": "requin-marteau"}},
    # Additional Categories
    "Vehicles": {
        1000: {"label": "car", "type": "Sedan", "usage": "Transportation"},
        1001: {"label": "airplane", "type": "Passenger jet", "usage": "Air travel"},
        1002: {"label": "bicycle", "type": "Mountain bike", "usage": "Recreation and exercise"},
    },
    "Objects": {
        "Household Items": {
            2000: {"label": "chair", "material": "Wood or plastic"},
            2001: {"label": "table", "material": "Wood or metal"},
        },
        "Sports Equipment": {
            820: {"label": "volleyball", "usage": "Team sport"},
            821: {"label": "ping-pong ball", "usage": "Table tennis"},
        },
    },
    # External Links and Confidence Threshold
    9999: {
        "label": "generic label", 
        "info_url": "https://example.com/generic_label", 
        "confidence_threshold": 0.8
    }
}
