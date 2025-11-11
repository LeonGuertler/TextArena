import textarena as ta

env_id = 'QuantumTicTacToe-v0'
predefined_actions = [
    '[0,1]',
    '[0,4]',
    '[3,4]',
    '[3,6]',
    '[5,8]',
    '[1,3]',
    '[2,5]',
    '[5,8]',
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
