# python visualise_vision_result.py \
#     ~/Documents/workspace/inverse_graphics/output/January2026/noisy_image_data_long/scenario0/data/dataset.npz \
#     ~/Documents/workspace/inverse_graphics/output/February2026/sir_custom_proposal/scenario0/results/vision_results.npz \
#     --start 0 --end 901 --seg-mask
python visualise_vision_result.py \
    ~/Documents/workspace/inverse_graphics/output/January2026/noisy_image_data_long/scenario0/data/dataset.npz \
    ~/Documents/workspace/inverse_graphics/output/February2026/sir_systematic_resample/scenario0/results/vision_results.npz \
    --start 0 --end 901
python visualise_vision_result.py \
    ~/Documents/workspace/inverse_graphics/output/January2026/noisy_image_data_long/scenario0/data/dataset.npz \
    ~/Documents/workspace/inverse_graphics/output/February2026/sir_debug/scenario0/results/vision_results.npz \
    --start 0 --end 901
# python visualise_vision_result_dual.py \
#     ~/Documents/workspace/inverse_graphics/output/January2026/noisy_image_data_long/scenario0/data/dataset.npz \
#     ~/Documents/workspace/inverse_graphics/output/February2026/sir_debug/scenario0/results/vision_results.npz \
#     ~/Documents/workspace/inverse_graphics/output/February2026/sir_debug/scenario0/results/vision_results_rerun.npz \
#     --start 0 --end 901


