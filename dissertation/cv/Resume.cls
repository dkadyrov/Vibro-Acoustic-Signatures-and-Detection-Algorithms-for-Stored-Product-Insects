\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Resume}[6/6/2013 custom CV class]
\LoadClass[10pt,letterpaper]{article} % Font size and paper type
\RequirePackage{titlesec}
\usepackage[parfill]{parskip} % Remove paragraph indentation
\usepackage{array} % Required for boldface (\bf and \bfseries) tabular columns
\usepackage{ifthen} % Required for ifthenelse statements

\pagestyle{empty} % Suppress page numbers

\titleformat{\section}
  {\large\scshape\bfseries\raggedright}
  {}{0em}
  {}
  [\titlerule]

  \titleformat{\subsection}
  {\large\raggedright}
  {}{0em}
  {}


\newcommand{\name}[1]{
  \centerline{\Large\bfseries #1}
}

\newcommand{\contact}[4]{
  \centerline{#1 {\large\textperiodcentered} #2 {\large\textperiodcentered} #3
    {\large\textperiodcentered} #4 \ {\Large}}
}

\newcommand{\longcontact}[5]{
    \center{#1 \hfill {\large\textperiodcentered} \hfill #2 \hfill {\large\textperiodcentered} \hfill #3}
    \center{ {\Large} #4 \hfill {\Large} #5}

}



%----------------------------------------------------------------------------------------
%	SECTION FORMATTING
%----------------------------------------------------------------------------------------

% Defines the rSection environment for the large sections within the CV
\newenvironment{rSection}[1]{ % 1 input argument - section name
  \sectionskip
  \MakeUppercase{\bf #1} % Section title
  \sectionlineskip
  \hrule % Horizontal line
  \begin{list}{}{ % List for each individual item in the section
    \setlength{\leftmargin}{0em} % Margin within the section
    \vspace{-.25em}
  }
  \item[]
}{
  \end{list}
}

%----------------------------------------------------------------------------------------
%	WORK EXPERIENCE FORMATTING
%----------------------------------------------------------------------------------------

\newenvironment{rSubsection}[4]{ % 4 input arguments - company name, year(s) employed, job title and location
 {\bf #1} \hfill {#2} % Bold company name and date on the right
 \ifthenelse{\equal{#3}{}}{}{ % If the third argument is not specified, don't print the job title and location line
  \\
  {\em #3} \hfill {\em #4} % Italic job title and location
  }\vspace{.20mm}
  \begin{list}{$\cdot$}{\leftmargin=0em} % \cdot used for bullets, no indentation
   \itemsep -.5em \vspace{-0.5em} % Compress items in list together for aesthetics
  }{
  \end{list}
  \vspace{-.250em} % Some space after the list of bullet points
}

% The below commands define the whitespace after certain things in the document - they can be \smallskip, \medskip or \bigskip
\def\namesize{\huge} % Size of the name at the top of the document
\def\addressskip{\smallskip} % The space between the two address (or phone/email) lines
\def\sectionlineskip{\smallskip} % The space above the horizontal line for each section
\def\nameskip{\bigskip} % The space after your name at the top
\def\sectionskip{\vspace{.5mm}} % The space after the heading section
