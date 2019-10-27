Dashboards with Tiki Hut
========================

Simply add ``"tiki_hut"`` to the list of training callbacks:

::

    tiki.Trainer().train(
        ...
        callbacks=["tiki_hut"],
        ...
    )

Start up the web server:

::

    $ tiki hut
