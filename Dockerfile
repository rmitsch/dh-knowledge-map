FROM continuumio/miniconda3

##########################################
# 1. Copy setup files into container.
##########################################

# Copy file with python requirements into container.
COPY environment.yml /tmp/environment.yml

##########################################
# 2. Install dependencies.
##########################################

ENV PATH /opt/conda/envs/tale/bin:$PATH
RUN apt-get update && \
    # Install system dependencies.
    apt-get -y --no-install-recommends install gcc g++ apt-utils make cmake nano && \
    # Install conda/pip depencencies; configure default environment.
    conda env update -n base -f /tmp/environment.yml

##########################################
# 3. Copy code.
##########################################

COPY . .

##########################################
# 4. Further setup
##########################################

# Expose and launch only if this is supposed to run frontend.
EXPOSE 8050