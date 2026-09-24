# Super Bowl XLIX: Lynch gets the ball (with narrative adjustments)  
  
## Part 1: the game  
  
NE at SEA (2014_21_NE_SEA), play 4205  
  real play : (:26) (Shotgun) 3-R.Wilson pass short right intended for 83-R.Lockette INTERCEPTED by 21-M.Butler at NE -1. 21-M.Butler to NE 2 for 3 yards (83-R.Lockette). PENALTY on NE, Unsportsmanlike Conduct, 1 yard, enforced at NE 2.  
  before    : SEA 24 - NE 28 | Q4 0:26 | SEA ball, 2&1 at NE 1 | timeouts SEA:1 NE:2  
  edit      : run  
  after     : SEA 24 - NE 28 | Q4 0:26 | SEA ball, 2&1 at NE 1 (next play: run) | timeouts SEA:1 NE:2  
  
Win probability after the edited play (10,000 simulations):  
  SEA   79.1%  (95% CI 78.3%-79.9%)  
  NE    20.9%  (95% CI 20.1%-21.7%)  
  
For comparison (SEA win probability):  
  nflfastR before the snap  63.3% | after the real play   6.0%  
  this model before snap    78.0% | after the real play   0.0%  
  real final: NE 28, SEA 24  
  
## Part 2: the next seasons  
  
10,000 simulated histories per world. Elo: K=20, HFA=48, regression=0.50, playoff x1.  
Ratings just before the changed game: SEA 1763, NE 1716  
  
Narrative adjustments:  
  SEA +15 Elo from 2015 week 1 (alternate, only if SEA won the changed game): No offseason hangover: the locker-room rift over the play call never happens  
  SEA +20 Elo from 2016 week 1 (alternate): Marshawn Lynch doesn't retire after 2015  
  NE -10 Elo from 2015 week 1 (alternate, only if SEA won the changed game): A Super Bowl loss adds to the Deflategate distraction  
  
season team                   Wins    Playoffs    Division Conf. title  Super Bowl  
----------------------------------------------------------------------------------  
2014   SEA   actual           12.0         yes         yes         yes           -  
             no change        12.0      100.0%      100.0%      100.0%        0.0%  
             with edit        12.0      100.0%      100.0%      100.0%       79.1%  
2014   NE    actual           12.0         yes         yes         yes         yes  
             no change        12.0      100.0%      100.0%      100.0%      100.0%  
             with edit        12.0      100.0%      100.0%      100.0%       20.9%  
2015   SEA   actual           10.0         yes           -           -           -  
             no change        10.3       68.2%       50.5%       19.5%       11.1%  
             with edit        10.8       74.1%       56.7%       23.2%       13.5%  
2015   NE    actual           12.0         yes         yes           -           -  
             no change        10.5       70.5%       51.6%       18.6%       10.5%  
             with edit        10.1       65.2%       46.6%       15.6%        8.4%  
2016   SEA   actual           10.5         yes         yes           -           -  
             no change         9.2       53.2%       37.0%       11.6%        6.4%  
             with edit         9.9       61.8%       44.0%       16.0%        9.2%  
2016   NE    actual           14.0         yes         yes         yes         yes  
             no change         9.1       52.7%       38.0%       10.9%        6.2%  
             with edit         8.9       49.9%       36.2%       10.0%        5.5%  
  
Other teams whose playoff odds moved most:  
  2015 BUF   43.7% ->  45.3% (+1.6%)  
  2015 SF    32.6% ->  31.1% (-1.5%)  
  2015 ARI   37.3% ->  35.8% (-1.5%)  
  2015 MIA   29.0% ->  30.3% (+1.4%)  
  2015 NYJ   23.0% ->  24.1% (+1.1%)  
  2016 ARI   39.3% ->  37.2% (-2.0%)  
  2016 SF    36.3% ->  34.6% (-1.7%)  
  2016 LA    30.0% ->  28.5% (-1.6%)  
  2016 GB    49.2% ->  48.2% (-1.1%)  
  2016 CAR   43.3% ->  42.4% (-0.9%)
