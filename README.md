# Curiosity
## General Variables, Data, and constant definitions

### `dict_words_boxes`
```
{img_id:(str): {
    obj_id:(str): {
        'bnbox': a list of four coordinations,
        'word':  the linguistic representation of the object
    },
},
```

Regular data (dictionary with all images and their object ids, corresponding words).

### `language_input`

a tensor of word indexes.
## `main.py`
### input
The shape of input to main.py

**< Number of Batches: int > < Learning Rate: float > < Curiosity Setting: str > < Seed >**

* Number of Batches: int, (default= 40)
* Learning Rate: [.1, .01, .001, .0001, .00001] : float
* Curiosity Setting: ["curious", "plasticity", "sn", "random"], (sn: Subjective Novelty)
* Seed: The ones used to run the experiment: [123, 234, 345, 456, 567, 678, 789, 890, 901, 12, 23, 34, 45, 56, 67, 78, 89, 90, 1, 100]

## Code Structure considerations
### Constants
In order to manage constants I made the class `UniversalConstants` for constants in `toolbox.py`. All the code files should import this class and make an object of it with specific name `uc`.
Here is how you do it:
```python
from toolbox import UniversalConstants
uc = UniversalConstants()
```

# Bug Problems
report bugs either here on GitHub or [contact Alireza](https://ali.mk/contact) personally.
