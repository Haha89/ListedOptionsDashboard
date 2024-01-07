export class Option {
    name ?: string
    type? : string
    strike?: number
    underlying? : string
    maturity? : Date
    price? : number
    delta?: number
    gamma?: number
    vega? : number
    theta?: number
    rho?: number
    implied_vol? : number

}

export interface SurfaceElement {
    strike: number
    maturity: string
    value: number
}