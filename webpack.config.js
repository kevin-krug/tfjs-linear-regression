const path = require("path");

module.exports = {
  mode: "development",
  entry: "./src/index.ts",
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  output: {
    filename: "main.js",
    path: path.resolve(__dirname, "dist"),
    library: "main",
  },
  devServer: {
    watchFiles: ["src/*"],
    static: {
      directory: path.join(__dirname, "dist"),
    },
    compress: false,
    port: 9000,
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
    ],
  },
};
