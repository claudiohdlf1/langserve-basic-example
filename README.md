# langserve-demo

## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

# Set your environment variables
create a .env file in the root of the project and add the following:

```
OPENAI_API_KEY=<your-api-key>
```

# Launch LangServe

## Launch LangServe
on the terminal run, on the poetry shell, run the following command:

```bash
poetry run langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
