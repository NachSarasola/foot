# Data Dictionary

This document describes the fields contained in the synthetic dataset located in the `data/` directory.

## events.csv

| Field | Type | Description |
| --- | --- | --- |
| `event_id` | int | Unique identifier for the event. |
| `match_id` | int | Identifier of the match (`matches.csv`). |
| `team_id` | int | Team performing the action (`teams.csv`). |
| `player_id` | int | Player responsible for the action (`players.csv`). |
| `period` | int | Match half: 1=first, 2=second. |
| `minute` | int | Minute of the match (0–90). |
| `second` | int | Second within the minute (0–59). |
| `event_type` | str | General classification (pass, carry, shot). |
| `x` | float | Start X coordinate on a 120×80 pitch. |
| `y` | float | Start Y coordinate on a 120×80 pitch. |
| `end_x` | float | End X coordinate. |
| `end_y` | float | End Y coordinate. |
| `outcome` | str | Result of the action (`complete` or `incomplete`). |
| `is_shot` | int | 1 if the event is a shot, else 0. |
| `is_goal` | int | 1 if the shot results in a goal. |
| `xg` | float | Expected goals value for shots (0–1). |
| `pass_switch` | int | 1 for long switches of play. |
| `carry_progressive` | int | 1 for carries that move the ball significantly forward. |
| `set_piece_type` | str | Type of set piece (`corner`, `free_kick`, `throw_in`, `penalty`, or empty for open play). |

## matches.csv

| Field | Type | Description |
| --- | --- | --- |
| `match_id` | int | Unique identifier of the match. |
| `date` | date | Match date. |
| `home_team_id` | int | Identifier of the home team. |
| `away_team_id` | int | Identifier of the away team. |
| `home_score` | int | Goals scored by the home team. |
| `away_score` | int | Goals scored by the away team. |
| `competition` | str | Competition name. |
| `season` | str | Season label. |

## players.csv

| Field | Type | Description |
| --- | --- | --- |
| `player_id` | int | Unique identifier of the player. |
| `team_id` | int | Team to which the player belongs. |
| `name` | str | Player name. |
| `position` | str | Primary position (`GK`, `DF`, `MF`, `FW`). |

## teams.csv

| Field | Type | Description |
| --- | --- | --- |
| `team_id` | int | Unique identifier of the team. |
| `name` | str | Team name. |
| `city` | str | Home city. |

## zones.csv

| Field | Type | Description |
| --- | --- | --- |
| `zone_id` | int | Unique zone identifier. |
| `name` | str | Zone label. |
| `x_min` | float | Minimum X boundary of the zone. |
| `x_max` | float | Maximum X boundary of the zone. |
| `y_min` | float | Minimum Y boundary of the zone. |
| `y_max` | float | Maximum Y boundary of the zone. |

