CALCLOUD MACHINE LEARNING DASHBOARD
======================

.. image:: https://readthedocs.org/projects/stsci-package-template/badge/?version=latest
    :target: https://stsci-package-template.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/spacetelescope/stsci-package-template/workflows/CI/badge.svg
    :target: https://github.com/spacetelescope/stsci-package-template/actions
    :alt: GitHub Actions CI Status

.. image:: https://codecov.io/gh/spacetelescope/stsci-package-template/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/spacetelescope/stsci-package-template
    :alt: Coverage Status

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


Interactive dashboard for analysis and evaluation of prediction models for CALCLOUD (Hubble Space Telescope's data reprocessing in the cloud package).

.. image:: previews/neural-network-graph.png
    :alt: Neural Network Interactive Prediction Tool


Run
-------

DOCKER

.. code-block:: bash
    
    docker pull alphasentaurii/calcloud-ml-dashboard:latest
    docker run -d -p 8050:8050 alphasentaurii/calcloud-ml-dashboard:latest


View the running dashboard in your browser: http://0.0.0.0:8050/


Install (for development)
-------

Using setup.py

.. code-block:: bash

    git clone https://grit.stsci.edu/rkein/calcloud-ml-dashboard
    cd calcloud-ml-dashboard
    python setup.py install --user


Using virtual env

.. code-block:: bash

    git clone https://grit.stsci.edu/rkein/calcloud-ml-dashboard
    python virtualenv dash-venv
    source dash-venv/bin/activate
    cd calcloud-ml-dashboard
    pip install -r requirements.txt


Once installed you can run the flask app locally

.. code-block:: bash
    
    git clone https://grit.stsci.edu/rkein/calcloud-ml-dashboard
    cd calcloud-ml-dashboard/calcloudML
    python app.py

View the running dashboard in your browser: http://127.0.0.1:8050/


Model Performance Evaluation
-------

Compare and evaluate model versions with roc-auc, precision-recall, keras history, and confusion matrix plots.

.. image:: previews/model-performance.png
    :alt: Accuracy vs Loss, Keras History 

Accuracy vs Loss Barplots and Keras History (train vs test)

.. image:: previews/roc-auc.png
    :alt: Receiver Operator Characteristic

Receiver Operator Characteristic (Area Under the Curve)


Exploratory Data Analysis
-------

Analyze data distributions, linearity and other characteristics.

.. image:: previews/eda-scatterplots.png
    :alt: Feature Scatterplots by Instrument

Feature Scatterplots by Instrument


.. image:: previews/eda-box-plots.png
    :alt: Feature Boxplots by Instrument

Feature Boxplots by Instrument

License
-------

See ``LICENSE.rst`` for more information.
