import { HttpClient } from "@angular/common/http"
import { Injectable } from "@angular/core"
import { Observable } from "rxjs"
import { Environment } from "src/assets/configs/environment"
import { Option } from "../models/option"
import { Surface, SurfaceRequest } from "src/models/surface"

@Injectable({ providedIn: 'root' })
export class OptionsService {

    constructor(private readonly http: HttpClient) { }


    getSpot(ticker: string): Observable<number> {
        return this.http.get<number>(Environment.baseUrl + '/spot/' + ticker)
    }

    getOptions(ticker: string, refresh = false, maturity: string): Observable<Option[]> {
        console.log(`Fetching listed options of ${ticker} ${refresh ? 'with' : 'without'} refresh`)
        //return this.http.get<Option[]>('../assets/fakedata/options.json')
        return this.http.get<Option[]>(Environment.baseUrl + '/options/' + ticker + '?refresh=' + refresh + "&maturity=" + maturity)
    }

    getSurface(ticker: string, optionType: string, field: string, range: number, maturity : string): Observable<Surface> {
        console.log(`Fetching option surface of ${optionType} ${ticker} for ${field}`)
        //return this.http.get<Surface>('../assets/fakedata/surface.json')
         return this.http.get<Surface>(`${Environment.baseUrl}/surface/${ticker}/${optionType}/${field}/${range}?maturity=${maturity}`)
    }

    postSurface(request : SurfaceRequest): Observable<Surface> {
        //return this.http.get<Surface>('../assets/fakedata/surface.json')
         return this.http.post<Surface>(`${Environment.baseUrl}/surface`, request)
    }
}