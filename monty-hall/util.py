import random


class Door(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "Door({value})".format(value=self.value)


class Game(object):

    def __init__(self, winning_door_name, options, default_choice=None):
        self.default_choice = default_choice if default_choice is not None else random.choice(list(options.keys()))
        self.winning_door_name = winning_door_name
        self.options = options

    def __repr__(self):
        return "Game Configuration:\n\t{doors}".format(
            doors="\n\t".join(["{} -> {}".format(name, door) for name, door in self.options.items()])
        )

    def end(self, select=None):
        if select is not None:
            return self.options[select]
        return self.options[self.default_choice]

    @staticmethod
    def random(number_of_options):
        door_names = list(range(number_of_options))
        winning_door_name = random.choice(door_names)
        options = {
            door_name: Door("goat") if door_name != winning_door_name else Door("car")
            for door_name in door_names
        }
        return Game(winning_door_name=winning_door_name, options=options)


class Host(object):

    def __init__(self, manage_game):
        self.game_evolution = [manage_game]

    def get_game(self, t):
        return self.game_evolution[t]

    def get_current_game(self):
        return self.get_game(t=-1)

    def show_options(self):
        return self.get_current_game().options

    def end_game(self, final_choice):
        return self.get_current_game().end(select=final_choice)

    def reveal(self, given_guest_choice):
        winning_door_name = self.get_current_game().winning_door_name
        reveal_tag = random.choice([
            door_name for door_name in self.get_current_game().options.keys()
            if door_name not in [winning_door_name, given_guest_choice]
        ])
        self.game_evolution.append(Game(
            winning_door_name=winning_door_name,
            options={
                name: door for name, door in self.get_current_game().options.items()
                if name != reveal_tag
            },
            default_choice=given_guest_choice
        ))
        return reveal_tag


def remember_function_output_decorator(func):
    pass


class Guest(object):

    class Strategy(object):
        RANDOM = "random"
        STAY = "stay"
        CHANGE = "change"
        all = [RANDOM, STAY, CHANGE]

    def __init__(self):
        self.memory = []

    def get_latest_choice(self):
        if not self.memory:
            raise ValueError("Guest doesn't has memory!")
        return self.memory[-1]

    def _choose_strategy_random(self, options):
        """Random Strategy
        The "random" strategy returns a random choice of all the available options.
        :param options: a dictionary containing door_name:door_obj
        :return: the key of the selected option (i.e., door_name).
        """
        option = random.choice([door_name for door_name in options.keys()])
        self.memory.append(option)
        return option

    def _choose_strategy_stay(self, options):
        """Stay Strategy
        The "stay" strategy return the previous choice.
        :param options: a dictionary containing door_name:door_obj
        :return: the key of the selected option (i.e., door_name)
        """
        option = self.get_latest_choice()
        self.memory.append(option)
        return option

    def _choose_strategy_change(self, options):
        """Change Strategy
        The "change" strategy returns any available option different than the current one (latest).
        :param options: a dictionary containing door_name:door_obj
        :return: the key of the selected option (i.e., door_name)
        """
        options.pop(self.memory[-1])
        select_key = self._choose_strategy_random(options=options)
        return select_key


    def choose(self, options, strategy):
        if strategy == Guest.Strategy.RANDOM:
            return self._choose_strategy_random(options=options)
        elif strategy == Guest.Strategy.STAY:
            return self._choose_strategy_stay(options=options)
        elif strategy == Guest.Strategy.CHANGE:
            return self._choose_strategy_change(options=options)
        else:
            raise ValueError("Not recognized strategy {}; Choose from {}".format(
                strategy, Guest.Strategy.all))


def play_random_game(number_of_options, strategy):
    # Game config
    game = Game.random(number_of_options=number_of_options)
    host = Host(manage_game=game)
    guest = Guest()
    # Game development
    first_choice = guest.choose(options=host.show_options(), strategy=Guest.Strategy.RANDOM)
    host.reveal(given_guest_choice=first_choice)
    final_choice = guest.choose(options=host.show_options(), strategy=strategy)
    result = host.end_game(final_choice=final_choice)
    return result.value
