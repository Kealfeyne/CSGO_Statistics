# Official tournament CS:GO matches analysis
## Data description
[Dataset](https://drive.google.com/file/d/1Zb-eDjJNv8ZStJ4oTXHtiLkcmt9V1Ypq/view?usp=sharing) contains of __108243__ entries with __106__ parameters. Matches in the dataset were played from __09.10.2012__ to __21.04.2022__.
### Match parameters:
- date - the date on which the match was played;
- tm1 - first team name;
- tm2 - second team name;
- map - the map on which the match was played;
- tm1pts - number of rounds the first team won;
- tm2pts - number of rounds the second team won;
- tm1side1pts - number of rounds the first team won in the first half of the game;
- tm2side1pts - number of rounds the second team won in the first half of the game;
- tm1side2pts - number of rounds the first team won in the second half of the game;
- tm2side2pts - number of rounds the second team won in the second half of the game;
- tm1rt - rating of the first team at the time of the match;
- tm2rt - rating of the second team at the time of the match;
- tm1fk - the number of first kills made by the first team;
- tm2fk - the number of first kills made by the second team;
- tm1clwon - number of clutches the first team won;
- tm2clwon - number of cluthes the second team won;
- The players are numbered from 1 to 10. Players 1-5 are the players of the first team, players 6-10 are the players of the second team. Each player has a set of parameters:
  + pl#nm - player nickname;
  + pl#kills - number of player`s kills;
  + pl#asts - number of player`s assists;
  + pl#deaths - number of player`s deaths;
  + pl#kast - percentage of rounds in which the player either had a kill, assist, survived or was traded;
  + pl#diffk-d - kills and deaths difference;
  + pl#adr - average damage per round;
  + pl#fkdiff - first kills difference;
  + pl#rt - player rating in the match, value from 0 to 2.
  
