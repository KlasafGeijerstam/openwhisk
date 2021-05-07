#!/usr/bin/fish

for i in (seq 1 400)
	wsk action invoke run_application --param-file ../symmetrical-telegram/apps/app10.json --result
	sleep 16
	curl localhost:8000/experiment_1 >> experiments/res_experiment_2_epoch_20
end
