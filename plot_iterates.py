def filter_data(xs, ys, gaps, ts, tol=0.003, en=lambda x: x):
    xs_filt = [xs[0]]
    ys_filt = [ys[0]]
    gaps_filt = [gaps[0]]
    ts_filt = [ts[0]]

    xrange = max([en(x) for x in xs]) - min([en(x) for x in xs])
    for (x, y, gap, t) in zip(xs, ys, gaps, ts):
        if abs(en(x)-en(xs_filt[-1]))/xrange + abs(y-ys_filt[-1]) > tol:
            xs_filt.append(x)
            ys_filt.append(y)
            gaps_filt.append(gap)
            ts_filt.append(t)
    print("Filtered out", len(xs) - len(xs_filt), "points out of", len(xs), "total.")
    
    
def generate(fig, outfile):
    if use_latex:
        with tempfile.TemporaryDirectory() as tmpdirname:
            print("Created temporary directory", tmpdirname)
            fig.savefig(tmpdirname + "/" + outfile, bbox_inches="tight")
            subprocess.run(["pdfcrop", tmpdirname + "/" + outfile, outfile]) 
    else:
        fig.savefig(outfile, bbox_inches="tight")
        
def make_plot(outfile, title, delta, eta, prox_fn, T=1000, omd=True, energy_plot=False, ylabel=False, inset=None):
    A = game_matrix(delta)
    xs, ys, gaps, ts = simulate(A, eta, prox_fn, T=T, omd=omd)

    for (t,x) in zip(ts, xs):
        if x >= 1/(1+delta):
            T1 = t
            break
    assert T1 is not None
    assert ts[T1] == T1

    for (t, y) in zip(ts, ys):
        if t > T1 and y >= 0.5/(1+delta):
            T2 = t
            break
    assert T2 is not None
    assert ts[T2] == T2

    T3 = None
    for (t, gap) in zip(ts, gaps):
        if t > T2 and y >= gap >= 0.1:
            T3 = t
            break
    # assert T3 is not None
    # assert ts[T3] == T3

    print("T1 is", T1, " and T2 is", T2)

    fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(.84*2.5 + (0.1 if ylabel else 0.0),.9*4.8))

    if energy_plot:
        en = lambda x: mpmath.log(x) - mpmath.log(1-x)
        ax1.set_xlabel(r"\!$\log(x^t[1]) - \log(1-x^t[1])$")
        if ylabel:
            ax1.set_ylabel("$y^t[1]$")
    else:
        en = lambda x: x
        ax1.set_xlabel("$x^t[1]$")
        if ylabel:
            ax1.set_ylabel("$y^t[1]$")

    # Filter so that we do not have too many points in the plot.
    xs_filt, ys_filt, gaps_filt, ts_filt = filter_data(xs, ys, gaps, ts, en=en)
    
    ax1.plot([en(x) for x in xs_filt], [y for y in ys_filt], 'o-', ms=.8, lw=.9)
    if inset == 1:
        axins = inset_axes(ax1, .4,.4, loc=2, bbox_to_anchor=(.595, 0.86),bbox_transform=ax1.figure.transFigure)
        axins.set_ylim((0.5-5e-2, 0.5+5e-2))
        axins_xs = []
        axins_ys = []
        for (x, y) in zip(xs_filt, ys_filt):
            axins_xs.append(-mp.log10(1-x))
            axins_ys.append(y)
        axins_xs = np.array(axins_xs)
        axins.plot(axins_xs[::3], axins_ys[::3], 'o-', ms=.3, lw=.5)
        axins.set_xlim((10, None))
        axins.tick_params(left = False, labelleft = False, bottom=False, labelbottom=False)
        # mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    elif inset == 2:
        axins = inset_axes(ax1, .4,.4, loc=2, bbox_to_anchor=(.57, 0.86),bbox_transform=ax1.figure.transFigure)
        axins.set_xlim((0.978, 1.002))
        axins.set_ylim((0.484, 0.508))
        axins_xs = []
        axins_ys = []
        axins.plot(xs_filt, ys_filt, 'o-', ms=.3, lw=.5)
        axins.tick_params(left = False, labelleft = False, bottom=False, labelbottom=False)
        # mark_inset(ax1, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    if inset:
        ax1.add_patch(patches.Rectangle((0.97, 0.45), 0.04, 0.08, fill=False, edgecolor='brown', lw=.5, zorder=1))
        axins.spines['left'].set_color('brown')
        axins.spines['right'].set_color('brown')
        axins.spines['top'].set_color('brown')
        axins.spines['bottom'].set_color('brown')
        ax1.plot([0.97, .765], [0.45, 0.485], c='brown', lw=.5)
        ax1.plot([0.97+.04, .924], [0.45+.08, 0.770], c='brown', lw=.5)

    ax1.plot([en(xs[T1])], [ys[T1]], 'r*', ms=7, mew=.5, mec='black')
    ax1.plot([en(xs[T2])], [ys[T2]], 'o', ms=5.5, mew=.5, mec='black', mfc='blue')
    if T3:
        ax1.plot([en(xs[T3])], [ys[T3]], 'gs', ms=5, mew=.5, mec='black',)
    ax1.set_ylim(0, 1)
    ax1.grid()
        
    ax2.loglog(ts_filt, gaps_filt, 'o-', ms=1, lw=1)
    ax2.loglog([T1], [gaps[T1]], 'r*', ms=7, mew=.5, mec='black')
    ax2.loglog([T2], [gaps[T2]], 'o', ms=5.5, mew=.5, mec='black', mfc='blue')
    if T3:
        ax2.plot([T3], [gaps[T3]], 'gs', ms=5, mew=.5, mec='black',)
    ax2.grid()
    ax2.set_xlabel("Iteration")
    if ylabel:
        ax2.set_ylabel("Equilibrium gap")

    ax1.set_title(title)
    fig.tight_layout()

    generate(fig, outfile)
    
def make_plot_best(outfile, title, delta, eta, prox_fn, T=1000, omd=True, energy_plot=False, ylabel=False, inset=None):
    A = game_matrix(delta)
    xs, ys, gaps, ts = simulate_best(A, eta, prox_fn, T=T, omd=omd)

    for (t,x) in zip(ts, xs):
        if x >= 1/(1+delta):
            T1 = t
            break
    assert T1 is not None
    assert ts[T1] == T1

    for (t, y) in zip(ts, ys):
        if t > T1 and y >= 0.5/(1+delta):
            T2 = t
            break
    assert T2 is not None
    assert ts[T2] == T2

    T3 = None
    for (t, gap) in zip(ts, gaps):
        if t > T2 and y >= gap >= 0.1:
            T3 = t
            break
    # assert T3 is not None
    # assert ts[T3] == T3

    print("T1 is", T1, " and T2 is", T2)

    fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(.84*2.5 + (0.1 if ylabel else 0.0),.9*4.8))

    if energy_plot:
        en = lambda x: mpmath.log(x) - mpmath.log(1-x)
        ax1.set_xlabel(r"\!$\log(x^t[1]) - \log(1-x^t[1])$")
        if ylabel:
            ax1.set_ylabel("$y^t[1]$")
    else:
        en = lambda x: x
        ax1.set_xlabel("$x^t[1]$")
        if ylabel:
            ax1.set_ylabel("$y^t[1]$")

    # Filter so that we do not have too many points in the plot.
    xs_filt, ys_filt, gaps_filt, ts_filt = filter_data(xs, ys, gaps, ts, en=en)
    
    ax1.plot([en(x) for x in xs_filt], [y for y in ys_filt], 'o-', ms=.8, lw=.9)
    if inset == 1:
        axins = inset_axes(ax1, .4,.4, loc=2, bbox_to_anchor=(.595, 0.86),bbox_transform=ax1.figure.transFigure)
        axins.set_ylim((0.5-5e-2, 0.5+5e-2))
        axins_xs = []
        axins_ys = []
        for (x, y) in zip(xs_filt, ys_filt):
            axins_xs.append(-mp.log10(1-x))
            axins_ys.append(y)
        axins_xs = np.array(axins_xs)
        axins.plot(axins_xs[::3], axins_ys[::3], 'o-', ms=.3, lw=.5)
        axins.set_xlim((10, None))
        axins.tick_params(left = False, labelleft = False, bottom=False, labelbottom=False)
        # mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    elif inset == 2:
        axins = inset_axes(ax1, .4,.4, loc=2, bbox_to_anchor=(.57, 0.86),bbox_transform=ax1.figure.transFigure)
        axins.set_xlim((0.978, 1.002))
        axins.set_ylim((0.484, 0.508))
        axins_xs = []
        axins_ys = []
        axins.plot(xs_filt, ys_filt, 'o-', ms=.3, lw=.5)
        axins.tick_params(left = False, labelleft = False, bottom=False, labelbottom=False)
        # mark_inset(ax1, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    if inset:
        ax1.add_patch(patches.Rectangle((0.97, 0.45), 0.04, 0.08, fill=False, edgecolor='brown', lw=.5, zorder=1))
        axins.spines['left'].set_color('brown')
        axins.spines['right'].set_color('brown')
        axins.spines['top'].set_color('brown')
        axins.spines['bottom'].set_color('brown')
        ax1.plot([0.97, .765], [0.45, 0.485], c='brown', lw=.5)
        ax1.plot([0.97+.04, .924], [0.45+.08, 0.770], c='brown', lw=.5)

    ax1.plot([en(xs[T1])], [ys[T1]], 'r*', ms=7, mew=.5, mec='black')
    ax1.plot([en(xs[T2])], [ys[T2]], 'o', ms=5.5, mew=.5, mec='black', mfc='blue')
    if T3:
        ax1.plot([en(xs[T3])], [ys[T3]], 'gs', ms=5, mew=.5, mec='black',)
    ax1.set_ylim(0, 1)
    ax1.grid()
        
    ax2.loglog(ts_filt, gaps_filt, 'o-', ms=1, lw=1)
    ax2.loglog([T1], [gaps[T1]], 'r*', ms=7, mew=.5, mec='black')
    ax2.loglog([T2], [gaps[T2]], 'o', ms=5.5, mew=.5, mec='black', mfc='blue')
    if T3:
        ax2.plot([T3], [gaps[T3]], 'gs', ms=5, mew=.5, mec='black',)
    ax2.grid()
    ax2.set_xlabel("Iteration")
    if ylabel:
        ax2.set_ylabel("Best Equilibrium gap")

    ax1.set_title(title)
    fig.tight_layout()

    generate(fig, outfile)
    
def make_plot_adagrad(outfile, title, delta, eps, prox_fn, T=1000, omd=True, energy_plot=False, ylabel=False, inset=None):
    A = game_matrix(delta)
    xs, ys, gaps, ts = simulate_adagrad(A, eps, prox_fn, T=T, omd=omd)

####################

    ### Commenting the T1, T2, T3 --- which may raise breaks
    # T1 = None
    # T2 = None
    # for (t,x) in zip(ts, xs):
    #     if x >= 1/(1+delta):
    #         T1 = t
    #         break
    # assert T1 is not None
    # assert ts[T1] == T1

    # for (t, y) in zip(ts, ys):
    #     if t > T1 and y >= 0.5/(1+delta):
    #         T2 = t
    #         break
    # assert T2 is not None
    # assert ts[T2] == T2

    # T3 = None
    # for (t, gap) in zip(ts, gaps):
    #     if t > T2 and y >= gap >= 0.1:
    #         T3 = t
    #         break
    # # assert T3 is not None
    # # assert ts[T3] == T3
    
    # print("T1 is", T1, "T2 is", T2)

    fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(.84*2.5 + (0.1 if ylabel else 0.0),.9*4.8))

    if energy_plot:
        en = lambda x: mpmath.log(x) - mpmath.log(1-x)
        ax1.set_xlabel(r"\!$\log(x^t[1]) - \log(1-x^t[1])$")
        if ylabel:
            ax1.set_ylabel("$y^t[1]$")
    else:
        en = lambda x: x
        ax1.set_xlabel("$x^t[1]$")
        if ylabel:
            ax1.set_ylabel("$y^t[1]$")

    # Filter so that we do not have too many points in the plot.
    xs_filt, ys_filt, gaps_filt, ts_filt = filter_data(xs, ys, gaps, ts, en=en)
    
    ax1.plot([en(x) for x in xs_filt], [y for y in ys_filt], 'o-', ms=.8, lw=.9)
    if inset == 1:
        axins = inset_axes(ax1, .4,.4, loc=2, bbox_to_anchor=(.595, 0.86),bbox_transform=ax1.figure.transFigure)
        axins.set_ylim((0.5-5e-2, 0.5+5e-2))
        axins_xs = []
        axins_ys = []
        for (x, y) in zip(xs_filt, ys_filt):
            axins_xs.append(-mp.log10(1-x))
            axins_ys.append(y)
        axins_xs = np.array(axins_xs)
        axins.plot(axins_xs[::3], axins_ys[::3], 'o-', ms=.3, lw=.5)
        axins.set_xlim((10, None))
        axins.tick_params(left = False, labelleft = False, bottom=False, labelbottom=False)
        # mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    elif inset == 2:
        axins = inset_axes(ax1, .4,.4, loc=2, bbox_to_anchor=(.57, 0.86),bbox_transform=ax1.figure.transFigure)
        axins.set_xlim((0.978, 1.002))
        axins.set_ylim((0.484, 0.508))
        axins_xs = []
        axins_ys = []
        axins.plot(xs_filt, ys_filt, 'o-', ms=.3, lw=.5)
        axins.tick_params(left = False, labelleft = False, bottom=False, labelbottom=False)
        # mark_inset(ax1, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    if inset:
        ax1.add_patch(patches.Rectangle((0.97, 0.45), 0.04, 0.08, fill=False, edgecolor='brown', lw=.5, zorder=1))
        axins.spines['left'].set_color('brown')
        axins.spines['right'].set_color('brown')
        axins.spines['top'].set_color('brown')
        axins.spines['bottom'].set_color('brown')
        ax1.plot([0.97, .765], [0.45, 0.485], c='brown', lw=.5)
        ax1.plot([0.97+.04, .924], [0.45+.08, 0.770], c='brown', lw=.5)

    # ax1.plot([en(xs[T1])], [ys[T1]], 'r*', ms=7, mew=.5, mec='black')
    # ax1.plot([en(xs[T2])], [ys[T2]], 'o', ms=5.5, mew=.5, mec='black', mfc='blue')
    # if T3:
    #     ax1.plot([en(xs[T3])], [ys[T3]], 'gs', ms=5, mew=.5, mec='black',)
    ax1.set_ylim(0, 1)
    ax1.grid()
        
    ax2.loglog(ts_filt, gaps_filt, 'o-', ms=1, lw=1)
    # ax2.loglog([T1], [gaps[T1]], 'r*', ms=7, mew=.5, mec='black')
    # ax2.loglog([T2], [gaps[T2]], 'o', ms=5.5, mew=.5, mec='black', mfc='blue')
    # if T3:
    #     ax2.plot([T3], [gaps[T3]], 'gs', ms=5, mew=.5, mec='black',)
    ax2.grid()
    ax2.set_xlabel("Iteration")
    if ylabel:
        ax2.set_ylabel("Equilibrium gap")
    
    ax1.set_title(title)
    fig.tight_layout()
    
    # Saving files locally
    generate(fig, f'{outfile}_eps_{eps}_delta_{delta}_T_{T}.pdf')