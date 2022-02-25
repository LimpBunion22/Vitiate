import matplotlib
import matplotlib.pyplot as plt

font = {#'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

COLORS = ['blue', 'orange', 'red', 'green']

def std_plot(x, y, args):
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)

    # Plot the requested variables
    lns = []

    if 'color' in args:
        color = args['color']
    else:
        color = 'blue'
    if 'label' in args:
        lns += ax.plot(x, y, color=('tab:'+color),
                        drawstyle='steps-post',
                        label=args['label'])
    else:
        lns += ax.plot(x, y, color=('tab:'+color),
                        drawstyle='steps-post')

    # Join all the plots
    # labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, bbox_to_anchor=[1.05, 1], loc='upper left')

    # Set plot title and labels
    if 'title' in args:
        plt.title(args['title'])
    if 'x_label' in args:
        ax.set_xlabel(args['x_label'])
    if 'y_label' in args:
        ax.set_ylabel(args['y_label'])

    # ax.set_xlim(0, results['time'][-1])

    fig.align_ylabels()
    fig.tight_layout()

    plt.draw()

    # Show all the plots in a single figure
    plt.show(block=False)

    return

def multiple_plot(x, y, args):
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)

    # Plot the requested variables
    lns = []

    ycont = 0
    for ylst in y:
        color = COLORS[ycont]
        if 'label' in args:
            lns += ax.plot(x, ylst, color=('tab:'+color),
                            drawstyle='steps-post',
                            label=args['label'][ycont])
        else:
            lns += ax.plot(x, ylst, color=('tab:'+color),
                            drawstyle='steps-post')
        ycont += 1

    # Join all the plots
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, bbox_to_anchor=[1.05, 1], loc='upper left')

    # Set plot title and labels
    if 'title' in args:
        plt.title(args['title'])
    if 'x_label' in args:
        ax.set_xlabel(args['x_label'])
    if 'y_label' in args:
        ax.set_ylabel(args['y_label'])

    # ax.set_xlim(0, results['time'][-1])

    fig.align_ylabels()
    fig.tight_layout()

    plt.draw()

    # Show all the plots in a single figure
    plt.show(block=False)

    return

def plot_and_wait(x, y, args):

    std_plot(x, y, args)
    input("Press any key to continue.")
    return

def multiple_plot_and_wait(x, y, args):

    multiple_plot(x, y, args)
    input("Press any key to continue.")
    return

def plot_images_from_set(set):

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.show(block=False)

    return