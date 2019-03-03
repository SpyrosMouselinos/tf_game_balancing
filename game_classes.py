from numpy.random import randint


class BaseClass(object):
    def __init__(self, hp=None, attack=None, defense=None, speed=None):
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.speed = speed
        self.current_hp = self.hp

    def set_atk(self, attack):
        self.attack = attack

    def set_def(self, defense):
        self.defense = defense

    def set_hp(self, hp):
        self.hp = hp

    def set_current_hp(self, hp):
        self.current_hp = hp

    def set_speed(self, speed):
        self.speed = speed

    def get_atk(self):
        return self.attack

    def get_def(self):
        return self.defense

    def get_hp(self):
        return self.hp

    def get_speed(self):
        return self.speed

    def get_current_hp(self):
        return self.current_hp

    def attack(self, opponent):
        if isinstance(opponent, BaseClass):
            damage = self.get_atk() - opponent.get_def()
            return opponent.set_current_hp(opponent.get_current_hp() - damage)
        else:
            raise(AttributeError, 'Attack should get a second argument of GameClass!\n')

    def is_dead(self):
        return self.get_current_hp() <= 0


class GameClass(BaseClass):
    def __init__(self, hp_range=None, attack_range=None, defense_range=None, speed_range=None, name=None):
        hp = randint(low=hp_range[0], high=hp_range[1])
        attack = randint(low=attack_range[0], high=attack_range[1])
        defense = randint(low=defense_range[0], high=defense_range[1])
        speed = randint(low=speed_range[0], high=speed_range[1])
        self.name = name
        super().__init__(hp=hp, attack=attack, defense=defense, speed=speed)

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name


class Warrior(GameClass):
    def __init__(self, hp_range=None, attack_range=None, defense_range=None, speed_range=None):
        super().__init__(hp_range=hp_range, attack_range=attack_range, defense_range=defense_range,
                         speed_range=speed_range, name='Warrior')

    @classmethod
    def make_random_warrior(cls, ranges):
        return cls(hp_range=ranges['hp'], attack_range=ranges['attack'], defense_range=ranges['defense'],
                   speed_range=ranges['speed'])


class Wizard(GameClass):
    def __init__(self, hp_range=None, attack_range=None, defense_range=None, speed_range=None):
        super().__init__(hp_range=hp_range, attack_range=attack_range, defense_range=defense_range,
                         speed_range=speed_range, name='Wizard')

    @classmethod
    def make_random_wizard(cls, ranges):
        return cls(hp_range=ranges['hp'], attack_range=ranges['attack'], defense_range=ranges['defense'],
                   speed_range=ranges['speed'])


class Rogue(GameClass):
    def __init__(self, hp_range=None, attack_range=None, defense_range=None, speed_range=None):
        super().__init__(hp_range=hp_range, attack_range=attack_range, defense_range=defense_range,
                         speed_range=speed_range, name='Rogue')

    @classmethod
    def make_random_rogue(cls, ranges):
        return cls(hp_range=ranges['hp'], attack_range=ranges['attack'], defense_range=ranges['defense'],
                   speed_range=ranges['speed'])
