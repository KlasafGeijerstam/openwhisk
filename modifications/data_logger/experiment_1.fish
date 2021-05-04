#!/usr/bin/fish

for i in (seq 1 600)
	wsk action invoke run_application --param-file ../symmetrical-telegram/apps/app10.json --result
	sleep 21
	curl localhost:8000/experiment_1 >> res_experiment_1.2_epoch_40
end
