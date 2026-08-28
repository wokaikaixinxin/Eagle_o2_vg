
submit_job --gpu 8 --tasks_per_node 1 --nodes 1 -n experiment --image /home/zhidingy/workspace/eagle2/torch2_test.sqsh \
        --logroot workdir_lasting_demo_short \
        --email_mode never \
        --partition adlr_services \
        --duration 0 \
        --dependent_clones 0 \
        -c "cd /home/zhidingy/workspace/eagle-video/streamlit_demo; bash start_demo.sh"
