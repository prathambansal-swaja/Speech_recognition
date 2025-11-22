
from common import (
    load_metadata, add_clip_lengths, plot_label_counts,
    filter_background_and_short, plot_clip_len_hist,
    print_key_percentiles, collect_example_signals,
    plot_signals, collect_noise_signals, plot_noise_signals,
    demo_resample
)

data = load_metadata()
data = add_clip_lengths(data)

plot_label_counts(data, "label", "Class")

data_no_noise = filter_background_and_short(data, max_len=1.1)
plot_clip_len_hist(data_no_noise)
print_key_percentiles(data_no_noise, [1, 3, 5, 7, 9, 10])

signals = collect_example_signals(data)
plot_signals(signals)
#plt.show()

noise_signals, titles = collect_noise_signals(data)
plot_noise_signals(noise_signals, titles)

demo_resample()
