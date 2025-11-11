import textarena as ta

env_id = 'Checkers-v0'
predefined_actions = [
    '[5 4 4 5]',
    '[2 1 3 0]',
    '[6 5 5 4]',
    '[2 7 3 6]',
    '[4 5 2 7]',
    '[3 0 4 1]',
    '[5 0 3 2]',
    '[2 3 4 1]',
    '[5 2 3 0]',
    '[1 0 2 1]',
    '[5 6 4 5]',
    '[1 2 2 3]',
    '[3 0 1 2]',
    '[0 3 2 1]',
    '[4 5 3 6]',
    '[2 5 4 7]',
    '[7 6 6 5]',
    '[0 1 1 2]',
    '[5 4 4 5]',
    '[2 3 3 4]',
    '[4 5 2 3]',
    '[1 4 3 2]',
    '[6 3 5 4]',
    '[3 2 4 1]',
    '[7 4 6 3]',
    '[4 1 5 0]',
    '[5 4 4 5]',
    '[0 5 1 4]',
    '[2 7 0 5]',
    '[2 1 3 0]',
    '[0 5 2 3]',
    '[1 2 3 4]',
    '[4 5 2 3]',
    '[4 7 5 6]',
    '[6 5 4 7]',
    '[0 7 1 6]',
    '[6 3 5 2]',
    '[1 6 2 7]',
    '[2 3 1 2]',
    '[3 0 4 1]',
    '[5 2 3 0]',
    '[2 7 3 6]',
    '[4 7 2 5]',
]


env = ta.make(env_id=env_id)
env.reset(num_players=2)

for istep, action in enumerate(predefined_actions):
    done, _ = env.step(action=action)
    if istep == len(predefined_actions) - 1:
        assert done, 'Game must end after the last move.'
    else:
        assert not done, 'Game must not yet end before the last move.'

print(f'{env_id=} test done successfully.')
