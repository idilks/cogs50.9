# cogs50.9: pomdps in healthcare
final project for cogs50.9, w25, idil sahin


 In order to construct a POMDP environment easily, [POMDP File Grammar](http://www.pomdp.org/code/pomdp-file-grammar.html) is used to encode environment dynamics.For the runner, solver and helper functions, credit to (Lu, 2018)(https://github.com/namoshizun/PyPOMDP)


positional arguments:
  config                The file name of algorithm configuration (without JSON
                        extension)

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             The name of environment's config file
  --budget BUDGET       The total action budget (defeault to inf)
  --snapshot SNAPSHOT   Whether to snapshot the belief tree after each episode
  --logfile LOGFILE     Logfile path
  --random_prior RANDOM_PRIOR
                        Whether or not to use a randomly generated
                        distribution as prior belief, default to False
  --max_play MAX_PLAY   Maximum number of play steps (episodes)

* example usage:
> python main.py pomcp --env pet-diagnosis.POMDP --budget 10
```
