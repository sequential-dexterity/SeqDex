declare -a tasks=( 'ShadowHandReOrientation' 'ShadowHandCatchAbreast' 'ShadowHandOver' 'ShadowHandBlockStack' 'ShadowHandCatchUnderarm'
'ShadowHandCatchOver2Underarm' 'ShadowHandTwoCatchUnderarm'
'ShadowHandDoorOpenInward' 'ShadowHandDoorOpenOutward' 'ShadowHandDoorCloseInward' 'ShadowHandDoorCloseOutward'
'ShadowHandPushBlock' 
'ShadowHandScissors' 'ShadowHandSwingCup' 'ShadowHandGraspAndPlace' 'ShadowHandSwitch' 'ShadowHandBottleCap' 'ShadowHandPen' 'ShadowHandPourWater' 'ShadowHandLiftUnderarm'
)

# for i in ${!tasks[@]}; do
# 	sudo rm -r /home/jmji/logs/tpami/distill/${tasks[$i]}/dapg/dapg_seed9 &
# done

# for i in ${!tasks[@]}; do
# 	cp -r /home/jmji/logs/tpami/distill/${tasks[$i]}/dapg/test_rew_3seeds.csv /home/jmji/logs/tpami/pc/logs/${tasks[$i]}/dapg/ &
# done

# declare -a lowertasks=("pour_water" "over" "catch_underarm" "catch_over2underarm" "catch_abreast" "two_catch_underarm" "push_block" 
# "re_orientation" "block_stack" "grasp_and_place" "door_open_inward" "door_open_outward" "door_close_inward" "door_close_outward" "switch" "lift_underarm"
# "pen" "swing_cup" "scissors" "bottle_cap")

# for i in ${!tasks[@]}; do
# 	cp -r /home/jmji/Downloads/data_merge/shadow_hand_${lowertasks[$i]}/ppo/test_rew_3seeds.csv /home/jmji/logs/tpami/pc/logs/${tasks[$i]}/ppo_origin &
# done

for i in ${!tasks[@]}; do
	cp -r /home/jmji/logs/tpami/pc/logs/${tasks[$i]}/${tasks[$i]}.png /home/jmji/logs/tpami/pc/figure_merge &
done
