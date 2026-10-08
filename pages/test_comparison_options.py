"""Player-comparison selector behaviour without requiring Streamlit."""

import pandas as pd
from src import comparison_options as comparison


def _demo_roster():
    rows = [
        ('A','U17','DEF'),('B','U17','DEF'),('C','U19','DEF'),
        ('D','U19','DEF'),('E','U17','MID'),('F','U17','MID'),
        ('G','U19','ATT')
    ]
    observations = pd.DataFrame(rows, columns=['player_id','squad','role'])
    return comparison.player_roster(observations)


def test_only_comparable_position_groups_are_shown():
    assert comparison.comparable_roles(_demo_roster()) == ['DEF', 'MID']


def test_squad_options_require_a_real_pair():
    assert comparison.comparable_squads(_demo_roster(), 'DEF') == ['U17', 'U19']
    assert comparison.comparable_squads(_demo_roster(), 'MID') == ['U17']


def test_default_scope_only_offers_same_squad_and_role():
    roster = _demo_roster()
    assert comparison.candidate_players(roster, 'DEF', 'U17') == ['A', 'B']
    assert comparison.candidate_players(roster, 'DEF', None) == ['A', 'B', 'C', 'D']


def test_second_player_never_repeats_first_player():
    eligible = comparison.candidate_players(_demo_roster(), 'DEF', 'U17')
    assert comparison.second_player_choices(eligible, 'A') == ['B']
    assert comparison.second_player_choices(eligible, 'B') == ['A']


def test_inconsistent_player_roster_is_rejected():
    invalid = pd.DataFrame([
        ('A','U17','DEF'),('A','U17','MID'),('B','U17','DEF')
    ], columns=['player_id','squad','role'])
    try:
        comparison.player_roster(invalid)
    except ValueError:
        return
    assert False, 'Conflicting player roles must not be accepted'
