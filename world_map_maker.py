# Create a world map
import numpy as np

def create_world_map(data, title, normalize=False, parts=[]):
    """Create a URL to a colorized world map with the google maps API

    Params:
    normalize- scale the input so that all the fractions for plotting
        fall between 0 and 1. This is useful if your input is not a
        fraction between 0 and 1

    Note: you need to open this in a browser to plot

    Note: we expect the data as dictionary where the keys are two
    letter country codes and the values are numbers that we should
    plot with the heat map

    """
    # convert the dict to a list of tuples with country code and value
    outputs = data.items()
    # sort by value
    outputs = sorted(outputs, key=lambda entry: entry[1])
    min_v, max_v = float(outputs[0][1]), float(outputs[-1][1])
    print min_v, max_v
    if not normalize:
        min_v, max_v = 0., 1.
    if parts != []:
        min_v, max_v = parts

    # partition the space depending on the number of colors
    buckets = np.array([int(float(x[1] - min_v)/max_v * 5) for x in outputs])
    outputs = zip(buckets, *zip(*outputs))
    # parts = np.linspace(min_v * 100., (max_v - min_v) * 100. / max_v, 5)
    parts = np.linspace(min_v * 100., max_v * 100., num=6)
    labels = []
    for index in range(1, 6):
        # num1 = int(parts[index - 1])
        # num2 = int(parts[index])
        labels.append("{:.1f}-{:.1f}%".format(parts[index - 1],
                                              parts[index]))

    # define our color map (10 colors at most)
    # colors = ["fff7ec", "fee8c8", "fdd49e", "fdbb84", "fc8d59",
    #           "ef6548", "d7301f", "b30000", "7f0000", "000000"]
    #
    # use this map because it is colorblind friendly and photocopy
    # safe (it should still print in grayscale)
    # colors = ["ffffcc", "a1dab4", "41b6c4", "2c7fb8", "253494"]
    # this colorscheme should also print in grayscale
    # colors = ["fef0d9", "fdcc8a", "fc8d59", "d7301f"]
    # colors = ["c6dbef", "9ecae1", "6baed6", "3182bd", "08519c", "eff3ff"]
    colors = ["c6dbef", "9ecae1", "6baed6", "3182bd", "08519c"]

    # put all the country arguments together
    map_args = ["http://chart.apis.google.com/chart?cht=map:"
                "fixed=-60,-20,80,-35", "chs=600x400", "chma=0,60,0,0"]
    # set all the unselected countries to be gray
    cnts, clrs = ["AA"], ["808080", "808080"]
    # setup the colors for the legend
    for color in colors:
        cnts.append("AA")
        clrs.append(color)
    # add the countries
    for (color, country, val) in outputs:
        cnts.append(country.upper())
        if color >= len(colors):
            color = len(colors) - 1
        clrs.append(colors[color])
    map_args.append("chld=" + "|".join(cnts))
    map_args.append("chco=" + "|".join(clrs))

    # add the legend
    map_args.append("chdl=No+Data|" + "|".join(labels))
    map_args.append("chdls=000000,14")
    # add the background to make the graph more visible
    map_args.append("chf=bg,s,EFF3FF")

    # create the title
    title = title.replace(" ", "+")
    map_args.append("chtt=" + title)
    map_args.append("chts=000000,20,c")

    print "&".join(map_args)

if __name__=='__main__':
    create_world_map({'CN':1, 'IN':0.5, 'US':0.01}, "try")
