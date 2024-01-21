export interface Surface {
    x: Date[]
    y: number[]
    z: number[][]
}

export class SurfaceRequest {
    ticker: string
    option_type: string
    field: string
    range: number
    maturity: string

    constructor(ticker: string, option_type: string, field: string, range: number, maturity: string) {
        this.field = field
        this.option_type = option_type
        this.ticker = ticker
        this.range = range
        this.maturity = maturity
    }
}