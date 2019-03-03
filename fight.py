import copy
from game_classes import *


def print_stats(x):
    print("{} Stats".format(x.get_name()))
    print("Attack {}".format(x.get_atk()))
    print("Defense {}".format(x.get_def()))
    print("Speed {}".format(x.get_speed()))
    print("Hp {}".format(x.get_current_hp()))
    return


def battle(x, y):
    _round = 1
    if y.get_speed() > x.get_speed():
        temp = copy.deepcopy(y)
        del y
        y = copy.deepcopy(x)
        del x
        x = copy.deepcopy(temp)
        del temp
    print("{} strikes first!".format(x.get_name()))
    print("Stats: \n")
    print_stats(x)
    print_stats(y)
    print("\n")
    while 1:
        x.fight(y)
        if y.is_dead():
            print("{} won!".format(x.get_name()))
            break
        y.fight(x)
        if x.is_dead():
            print("{} won!".format(y.get_name()))
            break
        print("Round {} results:\n".format(str(_round)))
        print("{} health : {}".format(x.get_name(), x.get_current_hp()))
        print("{} health : {}".format(y.get_name(), y.get_current_hp()))
        _round += 1
    print("Round {} results:\n".format(str(_round)))
    print("{} health : {}".format(x.get_name(), x.get_current_hp()))
    print("{} health : {}".format(y.get_name(), y.get_current_hp()))
    return


def main():
    x = Warrior().make_random(100)
    y = Wizard().make_random(100)
    battle(x, y)


if __name__ == '__main__':
    main()
