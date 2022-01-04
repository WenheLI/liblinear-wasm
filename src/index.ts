import LibLinearFactory, { LibLinearModule } from '../dist/liblinear';
let liblinear: LibLinearModule;

enum SOLVER_TYPE {
    L2R_LR = 0,
    L2R_L2LOSS_SVC_DUAL = 1,
    L2R_L2LOSS_SVC = 2,
    L2R_L1LOSS_SVC_DUAL = 3,
    SVC_CS = 4,
    L1R_L2LOSS_SVC = 5,
    L1R_LR = 6,
    L2R_LR_DUAL = 7,
    L2R_L2LOSS_SVR_DUAL = 11,
    L2R_L2LOSS_SVR = 12,
    L2R_L1LOSS_SVR_DUAL = 13,
    ONECLASS = 21
}

interface ILinearParam {
    solver_type?: SOLVER_TYPE;
    eps?: number;
    C?: number;
    nr_weight?: number;
    weight_label?: number[];
    weight?: number[];
    p?: number;
    nu?: number;
    init_sol?: number[];
    regularize_bias?: number;
    bias?: number;
}

class LinearParam {
    public param: ILinearParam = {
        solver_type: SOLVER_TYPE.L2R_LR,
        eps: 0.01,
        C: 1.0,
        nr_weight: 0,
        weight_label: [],
        weight: [],
        p: 0.1,
        nu: 0.5,
        init_sol: [],
        regularize_bias: 1,
        bias: -1
    };

    constructor(param?: ILinearParam) {
        if (param) {
            this.param = {
                ...this.param,
                ...param
            };
        }
    }
}

class Linear {
    private _modelPointer: number = -1;
    private _paramPointer: number = -1;
    public param: LinearParam;

    constructor(param?: ILinearParam) {
        this.param = new LinearParam(param);
    }

    public async init() {
        liblinear = await LibLinearFactory();
    }

    private checkInitialization = () => {
        if(!liblinear){
          throw new Error(`
          In order to use this SVM class, you'll need to initialize the linear (which grabs the WASM module and loads it asynchronously).
          Here's some example code.
          let linear = new Linear()
          await linear.init()
          linear.train(data)
          `)
        }
    }

    public train(data: number[][], label: number[]) {
        this.checkInitialization();
        const {
            solver_type,
            eps,
            C,
            nr_weight,
            weight_label,
            weight,
            p,
            nu,
            bias,
            init_sol,
            regularize_bias
        } = this.param.param;
        let rawWeightLabelPointer = 0;
        let rawWeightPointer = 0;
        let rawInitSolPointer = 0;

        if (weight_label) {
            const encodeWelghtLabel = new Int32Array(weight_label);
            rawWeightLabelPointer = liblinear._malloc(encodeWelghtLabel.length * 4);
            liblinear.HEAP32.set(encodeWelghtLabel, rawWeightLabelPointer / 4);
        }
        
        if (weight) {
            const encodeWeight = new Float64Array(weight);
            rawWeightPointer = liblinear._malloc(encodeWeight.length * 8);
            liblinear.HEAPF64.set(encodeWeight, rawWeightPointer / 8);
        }

        if (init_sol) {
            const initSol = new Float64Array(init_sol);
            rawInitSolPointer = liblinear._malloc(initSol.length * 8);
            liblinear.HEAPF64.set(initSol, rawInitSolPointer / 8);
        }

        this._paramPointer = liblinear._prepare_param(solver_type, C, eps, nr_weight, rawWeightLabelPointer, 
                                                      rawWeightPointer, p, 
                                                      nu, rawInitSolPointer, regularize_bias);
        const encodeData = new Float64Array(data.flat());
        const rawDataPointer = liblinear._malloc(encodeData.length * 8);
        liblinear.HEAPF64.set(encodeData, rawDataPointer / 8);

        const rawLabelPointer = liblinear._malloc(label.length * 8);
        const encodeLabel = new Float64Array(label);
        liblinear.HEAPF64.set(encodeLabel, rawLabelPointer / 8);

        const dataPointer = liblinear._init_problem(rawDataPointer, rawLabelPointer, data.length, data[0].length, bias);
        this._modelPointer = liblinear._train_model(dataPointer, this._paramPointer);

        liblinear._free_problem(dataPointer);
        liblinear._free(rawDataPointer);
        liblinear._free(rawLabelPointer);
        liblinear._free_param(this._paramPointer);
        this._paramPointer = -1;

        return this;
    }

    public predictOne(data: number[]): number {
        this.checkInitialization();
        const encodeData = new Float64Array(data);
        const rawDataPointer = liblinear._malloc(encodeData.length * 8);
        liblinear.HEAPF64.set(encodeData, rawDataPointer / 8);
        const result = liblinear._predict_one(this._modelPointer, rawDataPointer, data.length);
        liblinear._free(rawDataPointer);
        return result;
    }

    public predict(data: number[][]) {
        this.checkInitialization();
        const res = [];
        for (const d of data) {
            res.push(this.predictOne(d));
        }
        return res;
    }

    public predictProbability(data: number[][]) {
        this.checkInitialization();
        const res = [];
        for (const d of data) {
            const encodeData = new Float64Array(d);
            const rawDataPointer = liblinear._malloc(encodeData.length * 8);
            liblinear.HEAPF64.set(encodeData, rawDataPointer / 8);
            const resPointer = liblinear._predict_one_prob(this._modelPointer, rawDataPointer);
            const temp = Array.from(liblinear.HEAPF64.subarray(resPointer / 8, resPointer / 8 + d.length));
            res.push(temp);
            liblinear._free(resPointer);
            liblinear._free(rawDataPointer);
        }
        return res;
    }
}

export {
    Linear,
    LinearParam,
}