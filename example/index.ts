import { Linear } from "../src";

const main = async () => {
    const linear = new Linear();
    const data = [[1, 1], [-3, -4], [-5, -6], [5,6]];
    const label = [1, -1, -1, 1];
    await linear.init();
    linear.train(data, label);
    console.log(linear.predictProbability(data));
    for (let i = 0; i < data.length; i++) {
        console.log(`${data[i]} => ${label[i]}`);
    }
}

main();