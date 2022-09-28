from .ml43 import ML43Task

TRAIN_TASKS = [
"assembly-v2",
"basketball-v2",
"button-press-topdown-v2",
"button-press-topdown-wall-v2",
"button-press-v2",
"button-press-wall-v2",
"coffee-button-v2",
"coffee-pull-v2",
"coffee-push-v2",
"dial-turn-v2",
"disassemble-v2",
"door-close-v2",
"door-open-v2",
"drawer-close-v2",
"drawer-open-v2",
"faucet-open-v2",
"faucet-close-v2",
"hammer-v2",
"handle-press-side-v2",
"handle-press-v2",
"handle-pull-side-v2",
"lever-pull-v2",
"peg-insert-side-v2",
"pick-place-wall-v2",
"reach-v2",
"push-back-v2",
"push-v2",
"pick-place-v2",
"plate-slide-v2",
"plate-slide-side-v2",
"plate-slide-back-v2",
"plate-slide-back-side-v2",
"peg-unplug-side-v2",
"soccer-v2",
"stick-push-v2",
"stick-pull-v2",
"push-wall-v2",
"reach-wall-v2",
"shelf-place-v2",
"sweep-into-v2",
"sweep-v2",
"window-open-v2",
"window-close-v2",
]
TEST_TASKS = [
    "bin-picking-v2",
    "box-close-v2",
    "door-lock-v2",
    "door-unlock-v2",
    "hand-insert-v2",
]

train_tasks = [ ML43Task(n) for n in TRAIN_TASKS ]
test_tasks = [ ML43Task(n) for n in TEST_TASKS ]


class ML43Tasks:
    train_tasks = train_tasks
    test_tasks = test_tasks
